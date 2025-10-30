import os
import cv2
import joblib
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, Canvas, ttk, Scrollbar
from PIL import Image, ImageTk, ImageOps

# -------------------- CONFIGURAÇÕES --------------------
DATA_DIR = './Projeto/data'
MODEL_PATH = 'modelo_latas_aug.pkl'

# -------------------- LEITURA SEGURA --------------------
def safe_imread(path):
    try:
        img_pil = Image.open(path).convert("RGB")
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"[ERRO] Não foi possível abrir {path}: {e}")
        return None

# -------------------- EXTRAÇÃO DE FEATURES (FILTROS MANUAIS) --------------------
def aplicar_filtro(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2
    img_padded = np.pad(img, pad, mode='constant', constant_values=0)
    output = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            regiao = img_padded[i:i+kh, j:j+kw]
            valor = np.sum(regiao * kernel)
            output[i, j] = np.clip(valor, 0, 255)
    return output

def extrair_features_filtros(img):
    img = cv2.resize(img, (100, 100))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

    # Filtros
    filtro_media = np.ones((3,3))/9
    filtro_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filtro_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filtro_laplaciano = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    filtro_sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    media = aplicar_filtro(img_gray, filtro_media)
    sobel = aplicar_filtro(media, filtro_sobel_x) + aplicar_filtro(media, filtro_sobel_y)
    laplaciano = aplicar_filtro(media, filtro_laplaciano)
    sharpen = aplicar_filtro(media, filtro_sharpen)

    features = [
        np.mean(media), np.std(media),
        np.mean(sobel), np.std(sobel),
        np.mean(laplaciano), np.std(laplaciano),
        np.mean(sharpen), np.std(sharpen)
    ]
    return np.array(features)

# -------------------- DATA AUGMENTATION --------------------
def gerar_augmentations(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgs_aug = [
        np.array(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)),  # original
        np.array(cv2.cvtColor(np.array(pil_img.rotate(90)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(pil_img.rotate(180)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(pil_img.rotate(270)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(ImageOps.mirror(pil_img)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(ImageOps.flip(pil_img)), cv2.COLOR_RGB2BGR))
    ]
    return imgs_aug

# -------------------- TREINAMENTO --------------------
def treinar_modelo():
    try:
        label_status["text"] = "Treinando modelo... (aguarde)"
        root.update()

        features, labels = [], []
        image_names = []

        # Contar total de imagens para progress bar
        total_imgs = sum([len([f for f in os.listdir(os.path.join(DATA_DIR,d)) if f.lower().endswith(('.jpg','.png','.jpeg'))])
                          for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
        progress["maximum"] = total_imgs
        progress["value"] = 0

        for class_name in os.listdir(DATA_DIR):
            class_path = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(class_path):
                label_value = 0 if class_name.lower() == 'latas' else 1
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg','.png','.jpeg')):
                        img_path = os.path.join(class_path, filename)
                        img = safe_imread(img_path)
                        if img is not None:
                            for img_aug in gerar_augmentations(img):
                                feat = extrair_features_filtros(img_aug)
                                features.append(feat)
                                labels.append(label_value)
                                image_names.append(filename)
                            progress["value"] += 1
                            root.update()

        if len(features) < 10:
            messagebox.showerror("Erro", "Poucas amostras. Coloque imagens em 'data/latas' e 'data/outros'.")
            label_status["text"] = "Erro: poucas imagens."
            return

        X = np.array(features)
        y = np.array(labels)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test, train_names, test_names = train_test_split(
            X, y, image_names, test_size=0.2, random_state=50)

        model = SVC(kernel='linear', C=1.0, random_state=50)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

        joblib.dump((model, scaler), MODEL_PATH)

        # Mostrar dados treino/teste
        mostrar_dados_treino_teste(train_names, y_train, test_names, y_test)

        label_status["text"] = f"Modelo treinado com sucesso! Acurácia: {acc:.2f}%"
        messagebox.showinfo("Sucesso", f"Modelo treinado!\nAcurácia: {acc:.2f}%")

    except Exception as e:
        messagebox.showerror("Erro no treinamento", str(e))
        label_status["text"] = "Erro ao treinar modelo."

# -------------------- MOSTRAR DADOS TREINO/TESTE --------------------
def mostrar_dados_treino_teste(train_names, y_train, test_names, y_test):
    janela = Tk()
    janela.title("Dados de Treino e Teste")
    janela.geometry("500x400")

    frame = Frame(janela)
    frame.pack(fill='both', expand=True)

    # Scrollbar vertical
    canvas = Canvas(frame)
    scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas)
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    Label(scroll_frame, text="Treino:", font=("Arial",12,"bold")).pack()
    for n,lbl in zip(train_names, y_train):
        texto = f"{n} - {'LATA' if lbl==0 else 'OUTRO'}"
        Label(scroll_frame, text=texto).pack()

    Label(scroll_frame, text="--- Teste ---", font=("Arial",12,"bold")).pack(pady=5)
    for n,lbl in zip(test_names, y_test):
        texto = f"{n} - {'LATA' if lbl==0 else 'OUTRO'}"
        Label(scroll_frame, text=texto).pack()

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    janela.mainloop()

# -------------------- CARREGAR MODELO --------------------
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Erro", "Modelo não encontrado. Treine primeiro!")
        return None, None
    model, scaler = joblib.load(MODEL_PATH)
    return model, scaler

# -------------------- CLASSIFICAÇÃO --------------------
def classificar_imagem(img_path):
    model, scaler = carregar_modelo()
    if model is None:
        return

    img = safe_imread(img_path)
    if img is None:
        messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{img_path}")
        label_status["text"] = "Erro: imagem não aberta."
        return

    feat = extrair_features_filtros(img)
    feat = scaler.transform([feat])
    pred = model.predict(feat)[0]

    resultado = "LATA" if pred==0 else "OUTRO"
    label_status["text"] = f"Resultado: {resultado}"

    exibir_imagem(img_path, resultado)

# -------------------- INTERFACE --------------------
def selecionar_imagem():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if file_path:
        label_status["text"] = "Classificando imagem..."
        root.update()
        classificar_imagem(file_path)

def exibir_imagem(img_path, resultado_texto):
    img = Image.open(img_path)
    img = img.resize((250,250))
    tk_img = ImageTk.PhotoImage(img)

    canvas_imagem.create_image(0,0,anchor="nw",image=tk_img)
    canvas_imagem.image = tk_img
    label_resultado["text"] = resultado_texto

def thread_treinar():
    t = threading.Thread(target=treinar_modelo)
    t.start()

# -------------------- GUI --------------------
root = Tk()
root.title("Detector de Latas")
root.geometry("600x550")
root.resizable(False, False)

frame_top = Frame(root)
frame_top.pack(pady=10)

btn_treinar = Button(frame_top, text="Treinar Modelo", command=thread_treinar, bg="#4CAF50", fg="white", width=15)
btn_treinar.grid(row=0, column=0, padx=10)

btn_classificar = Button(frame_top, text="Selecionar Imagem", command=selecionar_imagem, bg="#2196F3", fg="white", width=15)
btn_classificar.grid(row=0, column=1, padx=10)

label_status = Label(root, text="Aguardando ação...", fg="gray", font=("Arial", 11))
label_status.pack(pady=5)

progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=5)

canvas_imagem = Canvas(root, width=250, height=250, bg="#EEE")
canvas_imagem.pack(pady=10)

label_resultado = Label(root, text="", font=("Arial",14,"bold"))
label_resultado.pack(pady=5)

root.mainloop()
