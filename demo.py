str='偶尔  有  老乡  拥  上来  想  看  “  大官  ”  ，  立即  会  遭到  “  闪开  ！'
# str='“  种菜  ，  也  有  烦恼  ，  那  是  累  的  时候  ；  另外  ，  大  棚  菜  在  降价  。'

def clean(s):
    if '“' not in s:  # 句子中间的引号不应去掉
        return s.replace(' ”', '')
    elif '”' not in s:
        return s.replace('“ ', '')
    elif '‘' not in s:
        return s.replace(' ’', '')
    elif '’' not in s:
        return s.replace('‘ ', '')
    else:
        return s
