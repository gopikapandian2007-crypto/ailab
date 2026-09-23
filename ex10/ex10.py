from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinominalNB
from sklearn.model_selection import train_test_split
data={
  "text":[
   "SyntaxError: invalid syntax",
   "IndentationError: unexpected indent",
   "SyntaxError: unexpected EOF while parsing",
   "ZeroDivisionError: division by zero",
   "IndexError: list index out of range",
   "NameError: name 'x' is not defined",
   "TypeError: unsupported operand type",
   "Output does not match expected result",
   "Loop runs infinite times, wrong condition used",
   "Wrong output due to incorrect formula used"
],
"label": [
   "Syntax Error", "Syntax Error", "Syntax Error",
   "Runtime Error", "Runtime Error", "Runtime Error", "Runtime Error",
   "Logical Error", "Logical Error", "Logical Error"
]
}
X_text = data["text"]
y = data["label"]
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(
 X,y,test_size=0.3, random_state=42
)
model=MultinominalNB()
model.fit(X_train, y_train)
def classify_error(error_message):
  vec = vectorizer.transform([error_message])
  prediction = model.predict(vec)
  return prediction[0]
test_errors = [
  "SyntaxError:missing colon",
  "TypeError: cannot add str and int",
  "Output is incorrect due to wrong loop condition"
]
for err in test_errors:
  print(f"Error:'{err}'--> Predicted Category: {classify_error(err)}")
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
