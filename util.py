import re

from numpy import NaN

def lint (str)->str:
    result = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]+').sub("",str if str is not None and str is not NaN else "")
    result = ' '.join(result.split())    
    return result