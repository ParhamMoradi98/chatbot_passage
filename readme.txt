This code is a chatbot that extracts the transcripts of a given Youtube channel.

User can ask questions about the content of the video.

For running this you need to install these python packages using this command: 

pip install yt-dlp sentence-transformers youtube-transcript-api faiss-cpu numpy Flask flask-cors Flask-Session python-dotenv requests


After installing the required packages,
1- Run Data_Base_Generation.ipynb  to generate the database given a Youtube channel url. 
2- Run app.py : This acts as the backend of the Chatbot. For production use cases, AWS can be used.
3- Open index.html: This is the front-end of the Chatbot. 

Working with Chatbot:
1- Write down something related to one of the videos ( for example type Martin Basiri) or a Youtube URL. 
Be aware this part is Case-sensitive.

Chatbot: The chatbot returns 5 related video to the written description

2- Select one of the videos by just typing its digit number (between 1,5)

3- Ask anything about that video. 
If your question is not related to the video, chatbot outputs :No relevant information found for this video. Try asking a different question.
 

4- At any stage you can type "exit" to start from part "1"

5- Type "exit" before closing the html file. 