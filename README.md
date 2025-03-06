<h1 style="font-family: san-serif; text-align: center; color: black; font-weight: 650;">📺 YouTube Comment Analysis</h1>

<h2 style="font-family: san-serif; text-align: left; color: #261324; font-weight: 550;">🎯 Purpose of the Application</h2>

<p style="font-family: san-serif; text-align: justify; color: #261324; font-weight: normal; font-size: 20px;">
  This application is designed for the automatic analysis of comments under YouTube videos. It allows users to easily retrieve, process, and examine comments, identifying key trends, emotional tone, and frequently used words. The tool is useful for bloggers, marketers, analysts, and anyone who wants to gain deeper insights into audience opinions.
</p>

<h2 style="font-family: san-serif; text-align: left; color: black; font-weight: 550;">📊 Sentiment Analysis</h2>

<p style="font-family: san-serif; text-align: justify; color: black; font-weight: normal; font-size: 20px;">
  Sentiment analysis is used to automatically determine the emotional tone of comments. Each comment can be classified as positive, negative, or neutral. It is important to note that this classification does not necessarily reflect the audience's attitude toward the video creator but rather indicates general public sentiment regarding the discussed topic. For example, negative comments may express emotions such as sadness, regret, sympathy, or dissatisfaction. Similarly, positive comments may convey joy, support, hope, or admiration.
</p>

<p style="font-family: san-serif; text-align: justify; color: black; font-weight: normal; font-size: 20px;">
  It is also important to consider that classification models may not always accurately recognize sarcastic or ironic statements. As a result, such messages may be mistakenly categorized into a different sentiment group, which should be considered when interpreting the analysis results.
</p>

<h2 style="font-family: san-serif; text-align: left; color: black; font-weight: 550;">🔍 Frequency Analysis</h2>

<p style="font-family: san-serif; text-align: justify; color: black; font-weight: normal; font-size: 20px;">
  Frequency analysis helps identify the most commonly used words in comments. This allows users to understand what topics or events are being most discussed by the audience. Through this analysis, users can quickly highlight key trends in comments and gain deeper insights into audience reactions or the topics being discussed in the video.
</p>

<h2 style="font-family: san-serif; text-align: left; color: black; font-weight: 550;">🔤 Supported Languages</h2>

<p style="font-family: san-serif; text-align: justify; color: black; font-weight: normal; font-size: 20px;">
  The application can analyze comments in 23 languages: English (en), Ukrainian (uk), French (fr), German (de), Italian (it), Japanese (ja), Korean (ko), Polish (pl), Portuguese (pt), Russian (ru), Spanish (es), Swedish (sv), Romanian (ro), Dutch (nl), Croatian (hr), Greek (el), Slovenian (sl), Norwegian Bokmål (nb), Macedonian (mk), Lithuanian (lt), Finnish (fi), Danish (da), Catalan (ca).
</p>

<h2 style="font-family: san-serif; text-align: left; color: black; font-weight: 550;">🤖 Model Used for Sentiment Analysis</h2>

<p style="font-family: san-serif; text-align: justify; color: black; font-weight: normal; font-size: 20px;">
  The sentiment analysis in this application is powered by the <a href="https://huggingface.co/cardiffnlp/xlm-roberta-base-sentiment-multilingual" target="_blank" style="color: #1E90FF; text-decoration: none;">cardiffnlp/xlm-roberta-base-sentiment-multilingual</a> model. This model is a multilingual variant of XLM-RoBERTa, fine-tuned to predict the sentiment of text in multiple languages. It classifies text into three sentiment categories: positive, negative, and neutral, making it suitable for analyzing YouTube comments in various languages.
</p>
