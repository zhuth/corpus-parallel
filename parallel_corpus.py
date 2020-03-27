#! /usr/bin/python3
# coding: utf-8
from functools import wraps
from flask import Flask, Response, render_template, jsonify, request, session, send_file, json
import os
import io
import re
import time
import datetime
from urllib.parse import quote as urlencode

from models import *


def pdfbase_encode(p):
    return f'pdffile={urlencode(p["pdffile"])}&pdfpage={p["pdfpage"]}'


def redirect(path):
    return render_template('redirect.html', path=path)


# COMMANDS
def add_user(uname, password):
    meta = get_meta()
    meta['users'].append(encrypt_password(uname, password))
    meta.save()


def align_titles(name):
    chnum = '[一二三四五六七八九十首甲乙丙丁戊己庚辛壬癸]'

    def roman(x):
        if '.' in x:
            x = x[:x.find('.')]
        return (',I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,' + x).split(',').index(x)

    def dechnum(x):
        ns = [(chnum+_).find(_) for _ in x]
        if len(ns) == 1:
            if ns[0] > 10:
                return ns[0] - 11
            else:
                return ns[0]
        elif len(ns) == 2:
            if ns[0] == 10:
                return ns[0] + ns[1]
            else:
                return -1
        else:
            return ns[0]*ns[1]+ns[2]

    def check_outline(content):
        outline = ''

        if lang == 'eng':
            lev, num, _ = (content + "  ").split(" ", 2)
            if lev == '§':
                outline = 'sect {:02}'.format(int(num[:-1]))
            elif lev.upper() == 'CHAPTER':
                outline = 'chap {:02}'.format(roman(num))
            elif lev.upper() == 'BOOK':
                outline = 'book {:02}'.format(roman(num))
            elif lev.upper().startswith('INTRODUCTION'):
                outline = 'book 00'
        else:
            if re.match('^' + chnum + '+、', content):
                outline = 'sect {:02}'.format(
                    dechnum(content[:content.find('、')]))
            elif re.match('^第' + chnum + '+章', content):
                outline = 'chap {:02}'.format(
                    dechnum(content[1:content.find('章')]))
            elif re.match('^第' + chnum + '+節', content):
                outline = 'sect {:02}'.format(
                    dechnum(content[1:content.find('節')]))
            elif re.match('^第' + chnum + '卷', content):
                outline = 'book {:02}'.format(
                    dechnum(content[1:content.find('卷')]))
            elif re.match('^篇' + chnum, content):
                outline = 'chap {:02}'.format(dechnum(content[1]))
            elif re.match('^部' + chnum, content):
                outline = 'book {:02}'.format(dechnum(content[1]))

        return outline

    for lang in ['chs', 'cht', 'eng']:

        nums = ['00', '00', '00']

        for p in Paragraph.query(F.collection == name):
            content = p.content.strip()
            if not content:
                continue

            outline = check_outline(content)

            if outline and outline[5] != '-':
                # print(content[:20], outline)
                if outline.startswith('book '):
                    nnums = [outline[5:], '00', '00']
                elif outline.startswith('chap '):
                    nnums = [nums[0], outline[5:], '00']
                else:
                    nnums = [nums[0], nums[1], outline[5:]]
                if '.'.join(nnums) > '.'.join(nums):
                    nums = nnums

            p.outline = '.'.join(nums)
            p.save()


def align_pdf(name, lang, pdfp):
    import fitz

    doc = fitz.open(pdfp)
    pages = doc.pageCount
    p = 0
    page_content = ''

    for paragraph in Paragraph.query(F.lang == lang):
        content = paragraph.content
        op = p
        kw = content[:5].split('，')[0]
        while kw not in page_content and p < pages - 1 and (op == 0 or p < op + 10):
            p += 1
            page_content = doc[p].getText().replace(' ', '').replace('\n', '')
        if p == op + 10 or p == pages - 1:
            print("ERR", content[:5], kw)
            p = op
        print(content[:5], p)
        paragraph.pdfsource = p
        paragraph.save()


def paragraph_finished(t):

    def _endswith(heap, needles):
        for _ in needles:
            if heap.endswith(_):
                return True
        return False

    return _endswith(t.strip(), '.!?…\"。！？…—：”')


def merge_lines(lines, lang):
    import statistics
    lens = [len(_) for _ in lines]
    if len(lens) < 3:
        yield ('' if lang[:2] == 'ch' else ' ').join(lines)
        return

    std = abs(statistics.stdev(lens))
    maxl = max(lens)
    t = ''
    last_line = '1'
    for l in lines:
        l = l.strip()
        if not l:
            continue
        if re.search(r'^[①-⑩]', l):
            break

        if lang[:2] != 'ch':
            t += ' '
        t += l
        if len(l) < maxl - std:
            if paragraph_finished(t) or not last_line:
                yield t
                t = ''
        last_line = l.strip()

    if t:
        yield t


def merge_paras(name):
    last_pdf = ''
    last_rid = 0
    last_page = ''
    accumulate_content = ''
    page = ''

    for p in Paragraph.query(F.collection == name).sort(F.lang, F.pdffile, F.pdfpage, F._id):
        content = re.sub(r'\d*(.+全集|部.|篇.)(（卷.）)?\s*\d*',
                         '', p.content.strip())
        if re.search(r'^[①-⑩]', content) or not content:
            p.delete()
            continue

        if last_page != p.pdfpage:
            page = ''
        if '①' in content:
            if '①' in page or content.find('①') != content.rfind('①'):
                content = content[:content.rfind('①')]
        page += content

        if p.lang not in ('cht', 'chs'):
            accumulate_content += ' '
        accumulate_content += content
        accumulate_content = accumulate_content.strip()

        if last_pdf != p.pdffile or paragraph_finished(accumulate_content):
            if accumulate_content != content and last_p != p:
                last_p.content = accumulate_content
                last_p.save()
                Paragraph.query((F._id > last_p.id) & (
                    F._id <= p.id) & (F.pdffile == last_pdf)).delete()
                print('merge paragraphs', last_p.id, 'through', p.id)
            accumulate_content = ''
        else:
            last_p = p

        last_pdf, last_page = p.pdffile, p.pdfpage


def set_yr(name):
    import yaml

    if not os.path.exists(f'dbs/{name}.yaml'):
        print('Specification file not found.')

    with open(f'dbs/{name}.yaml') as fy:
        y = yaml.safe_load(fy)
        fmt = y.get('format', r'.*{vol:02d}\.pdf')
        y = y['yr_align']
        for _y in y:
            if 'sty' in _y:
                yrs = range(_y['sty'], _y['edy']+1)
                pgs = [1] + _y['pgs'] + [10000-_y['offset']]
            else:
                yrs, pgs = [_[1] for _ in _y['pgs']], [_[0] for _ in _y['pgs']]
            offset = _y['offset'] - 1
            vol = _y['vol']
            for yr, ps, pe in zip(yrs, pgs, pgs[1:]):
                Paragraph.query(
                    (F.collection == name) &
                    (F.pdfpage >= ps+offset) & (F.pdfpage < pe+offset) &
                    F.pdffile.regex(fmt.format(vol=vol))
                ).update(_set={'year': yr})


def washout_duplicates(name):
    rids = []

    nc, np, npp = '', '', ''
    for p in Paragraph.query(F.collection == name).sort(F.pdffile, F.pdfpage):
        if nc == p.content and np == p.pdffile and npp == p.pdfpage:
            print(p.id)
            p.delete()
        nc, np, npp = p.content, p.pdffile, p.pdfpage


def washout_pagenum(name):

    # header
    for p in Paragraph.aggregator.match(F.collection == name).group(_id=[Var.pdffile, Var.pdfpage]).project(_first=1):
        paragraph_ = p.content

        if name == 'lx1':
            paragraph_ = re.sub(
                r'^([I\|](.{,3}年)+[I\|])?鲁迅.{,8}第[〇一二三四五六七八九十]+卷', '', paragraph_)
            paragraph_ = re.sub(
                r'^[\w\s]+[I丨\|]\s{,2}[〇一二三四五六七八九]+年\s{,2}[I丨\|]\s{,2}', '', paragraph_)
            paragraph_ = re.sub(r'^I', '', paragraph_)
            paragraph_ = re.sub(r'[〇oO0\s]+', '', paragraph_)

        elif name == 'scw':
            paragraph_ = re.sub(r'沈..全集◎', '', paragraph_)

        elif name == 'wgw':
            paragraph_ = re.sub(
                r'^...全集第.{,2}卷[一二三四五六七八九〇O0]*', '', paragraph_)

        p.content = paragraph_
        p.save()

    # footer
    for p in Paragraph.aggregator.match(F.collection == name).group(_id=[Var.pdffile, Var.pdfpage]).project(_last=1):
        paragraph_ = p.content

        if name == 'yf':
            if '26嚴復全集（卷一）' in paragraph_:
                print(paragraph_)
            paragraph_ = re.sub(r'..全集（.?卷[一二三四五六七八九十]*）\d+$', '', paragraph_)
            paragraph_ = re.sub(r'\d+..全集（.?卷[一二三四五六七八九十]*）$', '', paragraph_)

        else:
            paragraph_ = re.sub(r'[^\w]{,3}\d+[^\w]{,3}$', '', paragraph_)

        p.content = paragraph_
        p.save()


def import_pdf(name, lang, *files_or_patterns):
    import fitz
    import re
    import glob
    import os

    def __startswith(heap, needles):
        for n in needles:
            if heap.startswith(n):
                return True
        return False

    for filepattern in files_or_patterns:
        pdfs = glob.glob(filepattern)
        for _i, pdffile in enumerate(pdfs):
            pdffile_ = pdffile
            if pdffile.startswith('sources/'):
                pdffile = pdffile[len('sources/'):]
            if Paragraph.first(F.pdffile == pdffile):
                continue
            doc = fitz.open(pdffile_)
            pages = doc.pageCount
            print(pdffile, _i, len(pdfs), '         ', end='\r')

            para = ''
            outline = ''
            for p in range(pages):
                lines = doc[p].getText().split('\n')
                if name == 'wnl':
                    while lines and __startswith(lines[0], ('©', 'Wittgenstein\'s Nachlass', 'reserved.', 'Page: ')):
                        lines = lines[1:]

                    for l in lines:
                        if re.search(r'Item \d+[a-zA-Z]*\s+Page', l):
                            print(outline, end='\r')
                            P(name)(lang=lang, content=para, pdffile=pdffile, pdfpage=p,
                                    pagenum=p+1, collection=name, outline=outline).save()
                            # print(outline, para[:100], len(para))

                            outline = l.strip().split()
                            outline = '.'.join([outline[1], outline[-1]])
                            if pdffile == 'DIPLO.pdf':
                                outline = 'd.' + outline
                            para = ''
                            continue
                        para += l + ' '
                    if para and p == pages - 1:
                        P(name)(lang=lang, content=para, pdffile=pdffile, pdfpage=p,
                                pagenum=p+1, collection=name, outline=outline).save()
                    continue
                if lang[:2] == 'ch':
                    lines = [re.sub(r'([^a-zA-Z0-9\.\,]) ', r'\1', _)
                             for _ in lines]

                for para in merge_lines(lines, lang):
                    P(name)(lang=lang, content=para, pdffile=pdffile,
                            pdfpage=p, pagenum=p+1, collection=name).save()

    meta = get_meta()
    meta.pdffiles[name] = None
    meta.save()


def import_html(name, lang, *files):
    import zipfile
    from bs4 import BeautifulSoup as B

    def import_html_src(fname, html, outline=''):
        b = B(html, 'lxml')
        p = P(name)(
            lang=lang, content=b.text.strip(), pdffile=fname, pdfpage=0, pagenum=1,
            collection=name, outline=outline
        )
        p.content = str(b.find('body'))
        p.save()
        del b

    for f in files:
        if f.endswith('.zip'):
            with zipfile.ZipFile(f) as z:
                for f_ in z.filelist:
                    import_html_src(f, z.open(f_), f_.filename)
        else:
            with open(f, 'rb') as fin:
                import_html_src(f, fin)

    meta = get_meta()
    meta.pdffiles[name] = None
    meta.save()


def find_match(q):
    '''
    Query string `q` is a set of keywords (phrases) with options marked with ':'
    available options:
        `<fields>`: matching meta data
        `strict`: strict match
        `sort`: sort with keys
    '''

    def _split_query(q):
        quoted = ''
        kw = ''
        for k in q:
            if k in ("'", '"'):
                if quoted == k:
                    quoted = ''
                else:
                    quoted = k
            elif quoted:
                kw += k
            elif k == ' ':
                if kw:
                    yield kw
                kw = ''
            else:
                kw += k
        if kw:
            yield kw

    def _cut(kw):
        for k in re.findall(r'[\u4e00-\u9fff]+', kw):
            for w in jieba.cut(k):
                yield w
        for w in re.findall(r'[^\u4e00-\u9fff]+', kw):
            yield w.lower()

    def _kws(q):
        kws = set()
        opts = []
        for kw in _split_query(q):
            if ':' in kw:
                opts.append(kw.split(':', 1))
            elif kw.endswith('?'):
                opts.append(['r', '^' + kw[:-1]])
            else:
                kws = kws.union(set(_cut(kw)))
        return list(kws), opts

    kws, opts = _kws(q)

    cond = F.keywords.all(kws) if kws else MongoOperand({})

    optd = {}
    for optn, optv in opts:
        if optn == 'pdffile':
            cond &= F.pdffile == optv
        elif optn == 'pdfpage':
            cond &= F.pdfpage == int(optv)
        elif hasattr(Paragraph, optn[1:] if optn.startswith('!') else optn):
            if optn == 'outline':
                cond &= F[optn].regex('^' + optv.replace('.', r'\.'))
            elif optn.startswith('!'):
                cond &= F[optn[1:]] != optv
            else:
                cond &= F[optn] == optv
        elif optn == 'collections' and optv:
            cond &= F.collection.in_(optv.split(','))
        elif optn == 'r':
            cond &= F.keywords.regex(optv)
        elif optn == 're':
            cond &= F.content.regex(optv)
        elif optn:
            optd[optn] = optv
        elif optv == 'strict':  # strict mode
            optd['strict'] = [_ for _ in _split_query(q) if ':' not in _]
        elif optv == 'all':  # fetch all results
            optd['limit'] = 0

    rs = P(cond).query(cond)
    if 'sort' in optd:
        rs = rs.sort(*optd['sort'].split(','))
    if 'skip' in optd:
        rs = rs.skip(int(optd['skip']))

    def __strict_match(r):
        for k in optd['strict']:
            if k not in r['content']:
                return False
        return True

    if optd.get('limit') != 0:
        rs = rs.limit(int(optd.get('limit', 100)))

    results = rs.rs

    if 'strict' in optd:
        results = []
        for _ in rs.rs:
            if __strict_match(_):
                results.append(_)
                if len(results) == optd.get('limit', 100):
                    break

    return results, cond()


def export_xlsx(q, fn=''):
    q += ' :all'

    results, _ = find_match(q)

    import openpyxl
    from io import BytesIO
    wb = openpyxl.Workbook()
    ws = wb.active

    count = 0
    for rd in results:
        if count == 0:
            fields = [_ for _ in rd.keys() if _ != 'keywords']
            fields[0] = '#'
            ws.append(fields)
        count += 1
        ws.append([str(rd[_]) if _ != '#' else str(count) for _ in fields])

    if not fn:
        buf = BytesIO()
        wb.save(buf)
        return buf
    else:
        wb.save(fn)


# VIEWS


app = Flask(__name__)
app.secret_key = '!'


def require_login(func, goto='./login'):
    @wraps(func)
    def f(*args, **kwargs):
        if not session.get('login'):
            return redirect(goto)
        else:
            return func(*args, **kwargs)
    return f


@app.route('/login', methods=['POST', 'GET'])
def login_view():
    if request.form.get('u'):
        u = request.form.get('u')
        p = request.form.get('p')
        if encrypt_password(u, p) in get_meta()['users']:
            session['login'] = u
            return redirect('./')
        else:
            return redirect('./login')
    else:
        return render_template('login.html')


@app.route('/pdffile')
@require_login
def pdffiles_view():
    name = request.args.get('collection')
    r = pdffiles(name)
    return jsonify([{'pdffile': _, 'disp': _[:_.find('.')]} for _ in sorted(r)])


@app.route("/image")
@require_login
def page_image():
    pdffile, pdfpage = request.args.get('pdffile'), int(
        request.args.get('pdfpage', '0'))
    pdfpage += 1
    pdffile = f'sources/{pdffile}'.encode('utf-8')
    if not os.path.exists(pdffile):
        return 'Not found', 404

    from pdf2image import convert_from_path
    img = (convert_from_path(pdffile, 120, first_page=pdfpage,
                             last_page=pdfpage, fmt='png') or [None])[0]
    if img:
        buf = io.BytesIO()
        img.save(buf, format='png')
        buf.seek(0)
        return Response(buf, mimetype='image/png')
    else:
        return 'Err', 500


@app.route("/view")
@require_login
def show_content():
    name = request.args.get('collection')
    pdffile, pdfpage, outline = request.args.get('pdffile'), int(
        request.args.get('pdfpage', 0)), request.args.get('outline')

    corpus = {}
    cond = F.collection == name
    prev_para, next_para = '', ''

    if pdffile:
        prev_para = 'view?collection=' + name + '&' + pdfbase_encode(
            P(cond).query(
                cond & (F.pdffile < pdffile) | (
                    (F.pdffile == pdffile) & (F.pdfpage < pdfpage))
            ).sort(-F.pdffile, -F.pdfpage).first()
        )
        next_para = 'view?collection=' + name + '&' + pdfbase_encode(
            P(cond).query(
                cond & (F.pdffile > pdffile) | (
                    (F.pdffile == pdffile) & (F.pdfpage > pdfpage))
            ).sort(F.pdffile, F.pdfpage).first()
        )
        cond &= F.pdffile == pdffile
        cond &= F.pdfpage == pdfpage

    if outline:
        # | (F.outline.regex('^' + (outline + '.').replace('.', r'\.')))
        cond &= (F.outline == outline)
        prev_para = 'view?collection=' + name + '&outline=' + \
            P(cond).query(cond & (F.outline < outline)
                          ).sort(-F.outline).first().outline
        next_para = 'view?collection=' + name + '&outline=' + \
            P(cond).query(cond & (F.outline > outline + '.ZZ')
                          ).sort(F.outline).first().outline

    imgs = {}

    for p in P(cond).query(cond):
        if p.lang not in corpus:
            corpus[p.lang] = []
        corpus[p.lang].append(p)

        if p.lang not in imgs:
            imgs[p.lang] = '?' + pdfbase_encode(p)

    corpus = corpus.items()

    return render_template('view.html', **locals())


def safe_filename(q):
    return re.sub(r'[:?*/\\\.]', '_', q)


@app.route("/collections")
@require_login
def get_collections():
    meta = get_meta()
    return jsonify(meta.collections)


@app.route("/export", methods=["POST", "GET"])
@require_login
def export_view():
    q = request.args.get('q', request.form.get('q', ''))

    buf = export_xlsx(q)
    buf.seek(0)
    return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     attachment_filename=f'export {safe_filename(q)}.xlsx', as_attachment=True)


@app.route("/search", methods=["POST", "GET"])
@require_login
def search():
    q = request.args.get('q', request.form.get('q', ''))
    meta = get_meta()
    meta.history = (meta.history + [q])[-20:]
    meta.save()

    results, cond = find_match(q)
    results = list(results)
    for _ in results:
        _['pdfencoded'] = pdfbase_encode(_)

    return jsonify({'results': results, 'cond': cond})


@app.route("/change/<rid>/<oper>/<args>", methods=["POST", "GET"])
@app.route("/change/<rid>/<oper>", methods=["POST", "GET"])
@require_login
def operate(rid, oper, args=''):
    rid = ObjectId(rid)
    p = Paragraph.first(F.id == rid)

    args = args.split(',') if args else []

    if oper == 'delete':
        p.delete()
    if oper == 'merge':
        args = [ObjectId(_) for _ in args]
        p.content = ''.join(
            [_.content for _ in Paragraph.query(F._id.in_(args))])
        p.save()
    elif oper == 'edit':
        content = request.form['content'].strip()
        if content:
            p.content = content
            p.save()
        else:
            p.delete()
    else:
        return jsonify({'error': 'UNKNOWN OPERATION'})

    return jsonify({'message': 'OK'})


@app.route('/')
@app.route('/favicon.ico')
@require_login
def index_view():
    return render_template('index.html', login=session['login'])


@app.route('/wiki')
@require_login
def wiki_view():

    def search_wiki(kws):
        cond = MongoOperand({})

        if len(kws) == 1:
            r = Wikipedia.query(F.title == kws[0])
            if r.count(): return r

        for kw in kws:
            kw = kw.replace('\\', '\\\\').replace('.', r'\.').replace('*', '.*')
            if kw.startswith('-'): 
                cond &= MongoOperand({
                    'title': {
                        '$not': {
                            '$regex': kw[1:]
                        }
                    }
                })
            else:
                cond &= F.title.regex(kw)
        return Wikipedia.query(cond)
        
    wikis = search_wiki(request.args['q'].split()).limit(10)
    
    return render_template('wiki.html', wikis=wikis)


application = app


class JSON_Improved(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, bytes):
            return ''.join(['%02x' % _ for _ in o])
        if isinstance(o, ObjectId):
            return str(o)
        else:
            return json.JSONEncoder.default(self, o)


app.json_encoder = JSON_Improved


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        args = sys.argv[2:]
        if cmd in globals():
            a = globals()[cmd](*args)
            if a:
                print(a)
            exit()

    app.run(host='0.0.0.0', port=8371, debug=True)
