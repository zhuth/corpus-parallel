import jieba
from opencc import OpenCC
cc = OpenCC('t2s')

from PyMongoWrapper import *
from PyMongoWrapper.dbo import *
import config
dbo.connstr = 'mongodb://' + config.mongo + '/hamster'

# MODEL AND BASIC LOGIC

class Paragraph(DbObject):
    collection = str
    pdffile = str
    pdfpage = int
    keywords = list
    year = int
    outline = str
    content = str
    pagenum = int
    lang = str

    def __init__(self, lang='', content='', pdfsource='', **kwargs):
        super().__init__(**kwargs)

        if content:
            self.content = content
        if lang:
            self.lang = lang

        if lang == 'cht':
            content = cc.convert(content)

        if lang[:2] == 'ch':
            self.keywords = list(set(jieba.cut_for_search(content)))
        else:
            self.keywords = list(set(re.findall(r'\w+', content.lower())))

        if pdfsource:
            if ':' not in pdfsource:
                pdfsource += ':0000'
            self.pdffile, self.pdfpage = pdfsource.rsplit(':', 1)
            self.pdfpage = int(self.pdfpage)


class RMRB(Paragraph):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getattribute__(self, k):
        return super().__getattribute__(k)



class SLG(Paragraph):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getattribute__(self, k):
        return super().__getattribute__(k)


class HYJD(Paragraph):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getattribute__(self, k):
        return super().__getattribute__(k)


class Meta(DbObject):

    users = list
    history = list
    pdffiles = dict
    rootpath = str
    collections = list

    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, '')
        super().__init__(**kwargs)


class Wikipedia(DbObject):

    title = str
    wiki = str


def encrypt_password(u, p):
    from hashlib import sha1
    up = '{}_corpus_{}'.format(u, p).encode('utf-8')
    return '{}:{}'.format(u, sha1(up).hexdigest())


def P(cond):
    if isinstance(cond, MongoOperand):
        cond = cond()
    if isinstance(cond, dict):
        cond = cond.get('collection', '')
    if not isinstance(cond, str):
        return Paragraph
    return {
        'rmrb': RMRB,
        'slg': SLG,
        'hyjd': HYJD
    }.get(cond.lower(), Paragraph)


get_meta = lambda: Meta.first({})


def get_all_collections():
    return [_.collection for _ in Paragraph.aggregator.group(_id=Var.collection).project(collection=Var._id).perform()]


def pdffiles(name):
    if name in ('slg', 'rmrb', ''):
        return []
    
    meta = get_meta()
    res = meta.pdffiles.get(name)
    if not res:
        res = [_.pdffile for _ in P(name).aggregator.match(F.collection == name).group(_id=Var.pdffile).project(pdffile=Var._id).perform()]
        meta = get_meta()
        meta.pdffiles[name] = res
        meta.save()

    return res
