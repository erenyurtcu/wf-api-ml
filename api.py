import os
import time
from flask import Flask, request, jsonify, send_file
import pandas as pd
import re
import difflib
from collections import defaultdict
from openpyxl import load_workbook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_root(word):
    root = re.sub(r'(lu|lü|lı|li|sız|siz|de|da|ye|ya|e  |den|dan|la|le|te|ti|ta|u|i|ı|un|in|ün|siz|sız|suz|süz)$', '', word)
    return root

def is_valid_prediction(prediction, original_word):
    return len(prediction) > 3 and prediction.isalpha() and prediction in original_word

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        method = request.form.get("method")

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith('.xlsx'):
            return jsonify({"error": "Invalid file format"}), 400

        file.seek(0)
        try:
            wb = load_workbook(filename=file, data_only=True)
            sheet = wb.active
            data = sheet.values
            columns = next(data)
            df = pd.DataFrame(data, columns=columns)
        except Exception as e:
            return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 500

        if df.empty:
            return jsonify({"error": "The uploaded Excel file is empty."}), 400

        df['Cleaned'] = df[df.columns[0]].apply(lambda x: clean_text(str(x)))
        df['words'] = df['Cleaned'].apply(lambda x: x.split())

        root_mapping = {}
        for index, row in df.iterrows():
            for word in row['words']:
                root = extract_root(word)
                if root:
                    root_mapping[word] = root

        root_word_pairs = [(word, root_mapping[word]) for word in root_mapping]

        df_train = pd.DataFrame(root_word_pairs, columns=['word', 'root'])

        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        X = vectorizer.fit_transform(df_train['word'])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_train['root'])
        start_time = time.time()
        if method == "xgboost":
            model = XGBClassifier(eval_metric='mlogloss')
            model.fit(X, y)  # XGBoost modelini eğit
        elif method == "naivebayes":
            model = MultinomialNB()
            model.fit(X, y)  # Naive Bayes modelini eğit
        elif method == "svm":
            # SVM modelini eğitirken ve tahmin yaparken yoğun (dense) formatta kullanacağız
            model = SVC(kernel='linear')
            model.fit(X.toarray(), df_train['root'])  # SVM modelini eğit
        elif method == "ann":
            # ANN Modeli oluşturma ve eğitme
            X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

            ann_model = Sequential()
            ann_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
            ann_model.add(Dense(128, activation='relu'))
            ann_model.add(Dense(len(set(y)), activation='softmax'))

            ann_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

            ann_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

        else:
            return jsonify({"error": "Invalid method selected."}), 400

        def is_similar(predicted_root, original_word, threshold=0.6):
            similarity_ratio = difflib.SequenceMatcher(None, predicted_root, original_word).ratio()
            return similarity_ratio >= threshold

        def predict_root(text):
            cleaned_text = clean_text(text)
            words = cleaned_text.split()
            roots = []
            for word in words:
                if word:
                    if method == "svm":
                        prediction = model.predict(vectorizer.transform([word]).toarray())
                        predicted_root = prediction[0]
                    elif method == "ann":
                        prediction = ann_model.predict(vectorizer.transform([word]).toarray())
                        predicted_index = prediction.argmax(axis=1)[0]
                        predicted_root = label_encoder.inverse_transform([predicted_index])[0]
                    else:
                        prediction = model.predict(vectorizer.transform([word]))[0]
                        predicted_root = label_encoder.inverse_transform([prediction])[0]

                    # Tahmin edilen kökün geçerli olup olmadığını kontrol et
                    if is_valid_prediction(predicted_root, word) and is_similar(predicted_root, word):
                        roots.append(predicted_root)
                    else:
                        # Eğer tahmin edilen kök geçerli değilse, orijinal kelimeyi kök olarak kabul et
                        roots.append(word)
            return roots

        end_time = time.time()
        function_time = end_time - start_time
        word_row_dict = defaultdict(list)
        word_frequency = defaultdict(int)

        for index, row in df.iterrows():
            words = predict_root(row[df.columns[0]])
            for word in words:
                word_row_dict[word].append(index + 1)
                word_frequency[word] += 1

        filtered_word_row_dict = {word: rows for word, rows in word_row_dict.items() if word_frequency[word] >= 2}

        output_file = os.path.join(os.getcwd(), "result.txt")
        with open(output_file, "w") as f:
            for word, rows in filtered_word_row_dict.items():
                f.write(f"{word}: used in row(s) {', '.join(map(str, rows))}.\n")

        if os.path.exists(output_file):
            response = send_file(output_file, as_attachment=True)
            response.headers["X-Processing-Time"] = function_time
            return response
        else:
            return jsonify({"error": f"File {output_file} not found."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
