import os
import cv2
import joblib
import numpy as np
import threading
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

# -------------------- CONFIGURAÇÕES --------------------
DATA_DIR = './Projeto/data'
MODEL_PATH = 'modelo_latas_aug.pkl'
RESULTADOS_DIR = './resultados'
LOGO_PATH = 'Ecocan.png'
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# -------------------- FUNÇÕES BASE --------------------
def safe_imread(path):
    try:
        img_pil = Image.open(path).convert("RGB")
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"[ERRO] Não foi possível abrir {path}: {e}")
        return None

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

def segmentar_imagem(img_gray):
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    bordas = cv2.Canny(img_gray, 100, 200)
    return thresh, bordas

def extrair_features_filtros(img):
    img = cv2.resize(img, (100, 100))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtro_media = np.ones((3,3))/9
    filtro_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filtro_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filtro_laplaciano = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    filtro_sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    media = aplicar_filtro(img_gray, filtro_media)
    sobel = aplicar_filtro(media, filtro_sobel_x) + aplicar_filtro(media, filtro_sobel_y)
    laplaciano = aplicar_filtro(media, filtro_laplaciano)
    sharpen = aplicar_filtro(media, filtro_sharpen)
    thresh, bordas = segmentar_imagem(img_gray)
    features = [
        np.mean(media), np.std(media),
        np.mean(sobel), np.std(sobel),
        np.mean(laplaciano), np.std(laplaciano),
        np.mean(sharpen), np.std(sharpen),
        np.mean(thresh), np.std(thresh),
        np.mean(bordas), np.std(bordas)
    ]
    return np.array(features)

def gerar_augmentations(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgs_aug = [
        np.array(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)),
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
        label_status.configure(text="Treinando modelo... (aguarde)")
        app.update()
        features, labels = [], []
        total_imgs = sum([len([f for f in os.listdir(os.path.join(DATA_DIR,d)) if f.lower().endswith(('.jpg','.png','.jpeg'))])
                          for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
        progress.configure(maximum=total_imgs, value=0)
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
                            progress.step()
                            app.update()
        if len(features) < 10:
            messagebox.showerror("Erro", "Poucas amostras em 'data/latas' e 'data/outros'.")
            label_status.configure(text="Erro: poucas imagens.")
            return
        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', C=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        joblib.dump((model, scaler), MODEL_PATH)
        label_status.configure(text=f"Modelo treinado com sucesso! Acurácia: {acc:.2f}%")
        messagebox.showinfo("Sucesso", f"Modelo treinado!\nAcurácia: {acc:.2f}%")
    except Exception as e:
        messagebox.showerror("Erro no treinamento", str(e))
        label_status.configure(text="Erro ao treinar modelo.")

def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Erro", "Modelo não encontrado. Treine primeiro!")
        return None, None
    model, scaler = joblib.load(MODEL_PATH)
    return model, scaler

def classificar_imagem(img_path):
    model, scaler = carregar_modelo()
    if model is None:
        return
    img = safe_imread(img_path)
    feat = extrair_features_filtros(img)
    feat = scaler.transform([feat])
    pred = model.predict(feat)[0]
    resultado = "LATA" if pred == 0 else "OUTRO"
    label_status.configure(text=f"Resultado: {resultado}")
    exibir_imagem(img_path, resultado)

def selecionar_imagem():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if file_path:
        label_status.configure(text="Classificando imagem...")
        app.update()
        classificar_imagem(file_path)

def exibir_imagem(img_path, resultado_texto):
    img = Image.open(img_path)
    img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img)
    canvas_imagem.create_image(0, 0, anchor="nw", image=tk_img)
    canvas_imagem.image = tk_img
    label_resultado.configure(text=resultado_texto)

def visualizar_filtros():
    arquivos = [f for f in os.listdir(RESULTADOS_DIR) if f.endswith(".jpg")]
    if not arquivos:
        messagebox.showinfo("Aviso", "Nenhum filtro salvo ainda.")
        return
    for f in arquivos:
        img = cv2.imread(os.path.join(RESULTADOS_DIR, f))
        cv2.imshow(f, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thread_treinar():
    t = threading.Thread(target=treinar_modelo)
    t.start()

# -------------------- INTERFACE (TTKBOOTSTRAP) --------------------
app = tb.Window(themename="minty")
app.title("EcoCan - Tecnologia Sustentável")
app.geometry("650x700")
app.resizable(False, False)

# -------------------- LOGO ECOCAN --------------------
try:
    logo_path = os.path.join(os.getcwd(), LOGO_PATH)
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"Logo não encontrada em: {logo_path}")
    logo_img = Image.open(logo_path).convert("RGBA")
    logo_img = logo_img.resize((320, 220))
    logo_tk = ImageTk.PhotoImage(logo_img)
    lbl_logo = tb.Label(app, image=logo_tk)
    lbl_logo.image = logo_tk
    lbl_logo.pack(pady=(25, 10))
except Exception as e:
    print(f"[AVISO] Falha ao carregar logo: {e}")
    lbl_logo = tb.Label(app, text="EcoCan", font=("Segoe UI", 26, "bold"), bootstyle="success")
    lbl_logo.pack(pady=(25, 10))

# -------------------- TÍTULO --------------------
lbl_titulo = tb.Label(app, text="Tecnologia em prol da Reciclagem Sustentável",
                      font=("Segoe UI", 15, "bold"))
lbl_titulo.pack(pady=(0, 20))

# -------------------- BOTÕES VERDES ARREDONDADOS --------------------
frame_btns = tb.Frame(app)
frame_btns.pack(pady=9)

verde_principal = "#56cc9d"
verde_hover = "#D9F1DA"

def estilizar_botao(btn):
    btn.configure(
        bootstyle="success",
        width=15,
        cursor="hand2",
        style="Custom.TButton"
    )
    btn.bind("<Enter>", lambda e: btn.configure(style="Hover.TButton"))
    btn.bind("<Leave>", lambda e: btn.configure(style="Custom.TButton"))

# Criar estilo personalizado
style = tb.Style()
style.configure("Custom.TButton", background=verde_principal, foreground="white",
                font=("Segoe UI", 10, "bold"), borderwidth=0, relief="flat",
                focusthickness=3, focuscolor=verde_principal, padding=10, border=0)
style.map("Custom.TButton", relief=[("pressed", "flat")])
style.configure("Hover.TButton", background=verde_hover, foreground="white",
                font=("Segoe UI", 10, "bold"), padding=10)

btn_treinar = tb.Button(frame_btns, text="Treinar Modelo", command=thread_treinar)
btn_classificar = tb.Button(frame_btns, text="Selecionar Imagem", command=selecionar_imagem)
btn_filtros = tb.Button(frame_btns, text="Ver Filtros", command=visualizar_filtros)

for i, btn in enumerate([btn_treinar, btn_classificar, btn_filtros]):
    btn.grid(row=0, column=i, padx=10)
    estilizar_botao(btn)

# -------------------- MOLDURA IMAGEM VERDE --------------------
frame_img = tb.Frame(app, bootstyle="success", padding=8)
frame_img.pack(pady=20)

canvas_imagem = tb.Canvas(frame_img, width=250, height=250, background="#C8E6C9", highlightthickness=0)
canvas_imagem.pack()

# -------------------- RESULTADO / STATUS --------------------
label_resultado = tb.Label(app, text="", font=("Segoe UI", 14, "bold"))
label_resultado.pack(pady=(10, 5))

label_status = tb.Label(app, text="Aguardando ação...", font=("Segoe UI", 10))
label_status.pack(pady=(5, 10))

# -------------------- BARRA DE PROGRESSO --------------------
progress = tb.Progressbar(app, orient="horizontal", length=400, mode="determinate", bootstyle=SUCCESS)
progress.pack(pady=(5, 20))

app.mainloop()
