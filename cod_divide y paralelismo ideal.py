import csv
import threading
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Clase para el modelo de clasificación de sentimientos
class SentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = LogisticRegression()
        self.lock = threading.Lock()

    # Método para entrenar el modelo
    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    # Método para predecir el sentimiento de un mensaje
    def predict_sentiment(self, text, results):
        text_tfidf = self.vectorizer.transform([text])
        prediction = self.model.predict(text_tfidf)[0]
        with self.lock:
            results.append((text, prediction))  # Almacenar el mensaje y su predicción como una tupla

# Cargar datos del archivo CSV
def load_data(filename):
    X, y = [], []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Ignorar la primera línea si contiene encabezados
        for row in reader:
            X.append(row[1])  # Texto en la segunda columna
            y.append(row[3].upper())  # Emoción en la cuarta columna, convertida a mayúsculas
    return X, y

# Función para predecir sentimientos de un lote de mensajes de forma paralela
def predict_batch(classifier, messages, results, barrier):
    for msg in messages:
        classifier.predict_sentiment(msg, results)
        barrier.wait()

def calculate_work(user_messages, classifier):
    start_time = time.time()
    results = []
    for msg in user_messages:
        classifier.predict_sentiment(msg, results)
    end_time = time.time()
    work = end_time - start_time
    return work

def calculate_span(user_messages, classifier):
    start_time = time.time()

    results = []
    for msg in user_messages:
        classifier.predict_sentiment(msg, results)

    end_time = time.time()
    span = end_time - start_time
    return span

def calculate_parallelism_time(user_messages, classifier):
    # Calcular el trabajo
    work = calculate_work(user_messages, classifier)

    # Calcular el span
    span = calculate_span(user_messages, classifier)

    # Calcular el paralelismo ideal como work/span
    ideal_parallelism_time = work / span
    print("Paralelismo ideal:", ideal_parallelism_time)

# Función para dividir los mensajes en lotes y procesarlos en paralelo
def divide_and_conquer(messages, classifier, num_threads):
    batch_size = len(messages) // num_threads
    results = []
    barrier = threading.Barrier(num_threads)

    threads = []
    for i in range(num_threads):
        start_index = i * batch_size
        end_index = start_index + batch_size if i < num_threads - 1 else len(messages)
        batch_messages = messages[start_index:end_index]
        thread = threading.Thread(target=predict_batch, args=(classifier, batch_messages, results, barrier))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return results

def main(user_messages, classifier):
    # Calcular el paralelismo ideal
    calculate_parallelism_time(user_messages, classifier)

    # Dividir y conquistar: Procesar los mensajes en lotes en paralelo
    results = divide_and_conquer(user_messages, classifier, num_threads=2)

    # Visualización de los resultados
    emotions = defaultdict(int)
    for msg, result in results:
        print(f"Mensaje: {msg} - Predicción: {result}")
        emotions[result] += 1

    labels = list(emotions.keys())
    sizes = list(emotions.values())

    # Crear gráfico de barras
    plt.bar(labels, sizes)
    plt.xlabel('Sentimiento')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Sentimientos')
    plt.show()

if __name__ == "__main__":
    # Cargar datos del archivo CSV
    X, y = load_data("data.csv")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de clasificación de sentimientos
    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)

    # Pedir al usuario cuántos mensajes desea ingresar
    num_messages = int(input("¿Cuántos mensajes quieres ingresar? "))
    user_messages = [input(f"Introduce el mensaje {i + 1}: ") for i in range(num_messages)]

    main(user_messages, classifier)
