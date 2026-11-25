from transformers import pipeline

clf = pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")
for t in ["이번 교육 정말 유익했어요", "실습 환경 세팅이 너무 오래 걸렸어요"]:
    print(t, "->", clf(t)[0])

summ = pipeline("summarization", model="gogamza/kobart-summarization")
text = "사내 AI 정기 교육은 매주 목요일에 진행된다. 하반기에는 크롤링, RAG, Agent를 다룬다. 실습은 GitHub 저장소로 제출한다."
print(summ(text, max_length=40)[0]["summary_text"])
