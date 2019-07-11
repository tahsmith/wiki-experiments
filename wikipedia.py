import io
import pickle
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from glob import glob
from multiprocessing import Queue, Process

from lxml import etree


def main():
    queue = Queue()
    process = Process(target=write_pickles,
                      args=(queue, 'wiktionary_revisions.p',))
    process.start()
    revisions = get_revisions(find_stub_meta_history(), limit=-1)
    for item in revisions:
        queue.put(item)
    queue.put(None)
    process.join()

    dump_iterable('wiktionary_username_map.p', revisions.result)

    articles = collect_usernames_from_articles(load_iterable(
        'wiktionary_revisions.p'))

    dump_iterable('wiktionary_article_users_map.p', articles)


def find_stub_meta_history(wiki='*', date='*', root='./data'):
    return glob(root + f'/{wiki}-{date}-stub-meta-history.xml')[0]


class ReturningGenerator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.result = yield from self.gen


def returning_generator(gen):
    @wraps(gen)
    def f(*args, **kwargs):
        return ReturningGenerator(gen(*args, **kwargs))

    return f


@returning_generator
def get_revisions(filename, limit=-1):
    def match_log_item(elem):
        return elem.tag.endswith('page')

    fields = [
        ('title',
         optional_child('./mw:title')),
        ('revision_timestamp_list',
         list_of_children('./mw:revision/mw:timestamp')),
        ('contributor_id_list',
         list_of_children('./mw:revision/mw:contributor/mw:id')),
        ('contributor_username_list',
         list_of_children('./mw:revision/mw:contributor/mw:username')),
    ]

    count = 0
    username_map = defaultdict(set)
    with open(filename, 'rb') as f:
        for log_item in search_for_elems(f, match_log_item, fields):
            for id, name in zip(log_item['contributor_id_list'],
                                log_item['contributor_username_list']):
                username_map[id].add(name)
            yield log_item
            count += 1
            if count % 1000 == 0:
                print(log_item)
            if count == limit:
                break

    username_map = {k: list(v) for k, v in username_map.items()}

    return username_map


def collect_usernames_from_articles(articles):
    for article in articles:
        yield (article['title'], set(article['contributor_username_list']))


def get_current_wiki_titles(queue, filename):
    count = 0

    def match_page(elem):
        return elem.tag.endswith('page')

    fields = [
        ('title', optional_child('./mw:title'))
    ]

    with open(filename, 'rb') as f:
        for page in search_for_elems(f, match_page, fields):
            queue.put(page)
            count += 1
            if count % 1000 == 0:
                print(f'processed: {count:<12}')
    queue.put(None)


def search_for_elems(f, match_fn, children):
    for (event, elem) in etree.iterparse(f, events=('end',)):
        if not match_fn(elem):
            continue

        page = {
            k: f(elem)
            for k, f in children
        }
        yield page

        elem.clear()


def write_pickles(queue, filename):
    with open(filename, 'wb') as f:
        while True:
            page = queue.get()
            if page is None:
                break
            pickle.dump(page, f)


def optional_child(child):
    def f(elem):
        text = elem.findall(child, namespaces)
        if text and len(text) == 1:
            text = text[0]
            return text.text
        return None

    return f


def list_of_children(child):
    def f(elem):
        children = elem.findall(child, namespaces)
        return list(x.text for x in children)

    return f


def dumpf(f, obj):
    with file_or_open(f, 'wb') as f:
        pickle.dump(obj, f)


def loadf(f):
    with file_or_open(f, 'rb') as f:
        return pickle.load(f)


def dump_iterable(f, iterable):
    with file_or_open(f, 'wb') as f:
        for x in iterable:
            pickle.dump(x, f)


def load_iterable(f):
    with file_or_open(f, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


@contextmanager
def file_or_open(f, mode ):
    if isinstance(f, str):
        kwargs = {}
        if mode is not None:
            kwargs['mode'] = mode
        f = open(f, **kwargs)
        close = True
    elif isinstance(f, io.IOBase):
        close = False
    else:
        raise ValueError('f must be a string of a file like object. Got {type}'
                         .format(type=type(f)))

    try:
        yield f
    finally:
        if close:
            f.close()


namespaces = {
    'mw': 'http://www.mediawiki.org/xml/export-0.10/'
}

if __name__ == '__main__':
    main()
