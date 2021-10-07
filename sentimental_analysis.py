import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
nltk.download('stopwords')


def sentiments(review):
    corpus=[]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    with open(r"C:\Users\Anshul\Desktop\models.pkl","rb") as f:
        models=pickle.load(f)
        
    review_data=models["Vectorizer"].transform([review]).toarray() 
    pred=models["BNB"].predict(review_data)
    
    #return pred[0]
    if int(pred[0])==1:
        return "Positive"
    elif int(pred[0])==0:
        return "Negative"
