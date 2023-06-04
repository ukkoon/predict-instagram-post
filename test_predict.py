import re
from konlpy.tag import Hannanum
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle
from util import lint

hnn = Hannanum()

def sentiment_predict(model,new_sentence):
  with open('tokenizer', 'rb') as handle:
    tokenizer = pickle.load(handle)

  new_sentence = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]+').sub("",new_sentence)
  new_sentence = hnn.nouns(new_sentence)
  encoded = tokenizer.texts_to_sequences([new_sentence])
  pad_new = pad_sequences(encoded, maxlen = 100)

  score = float(model.predict(pad_new))
  if(score > 0.5):
    print("{:.2f}% 확률로 라플 포스트입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 논라플 포스트입니다.".format((1 - score) * 100))

model = load_model("./model",compile=True)
result = sentiment_predict(
    model,
    lint("[국내발매] JD스포츠에서 덩크 로우 범고래 GS 제품 응모가 시작되었습니다🔥 . 응모 링크는 @luckydraw_kr 프로필🔥 https://www.luck-d.com/release-raffle/detail/dunk-low-black-white-gs/ . 🚨 게시물 알림 설정 하시고 한정판 스니커즈 발매정보 놓치지 마세요😍 . 한정판 발매 정보의 모든것! ▶ APP : 럭키드로우 ▶ App download : https://luckydraw.page.link/2L6u ▶ Website : https://www.luck-d.com ▶ Instagram : @luckydraw_kr ▶ Facebook 페이지 : 럭키드로우 . #luckydraw_kr #Raffle #LuckyDraw #럭키드로우 #당첨 . #나이키 #NIKE #Dunk #덩크 #덩크로우 #Dunklow"),
)