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

# -------------------- CONFIGURAÇÕES GLOBAIS --------------------
DATA_DIR = './Projeto/data'
MODEL_PATH = 'modelo_latas_aug.pkl'
RESULTADOS_DIR = './resultados'
LOGO_PATH = 'Ecocan.png'
os.makedirs(RESULTADOS_DIR, exist_ok=True) # Garante que a pasta existe

# -------------------- FUNÇÕES BASE (Processamento de Imagem) --------------------

# Leitura de imagem mais segura que o cv2.imread puro
def safe_imread(path):
    try:
        # Tenta abrir com PIL (lida melhor com formatos e cores)
        img_pil = Image.open(path).convert("RGB")
        img = np.array(img_pil)
        # Converte PIL(RGB) -> OpenCV(BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"[ERRO] Não foi possível abrir {path}: {e}")
        return None

# Implementação manual da convolução 2D
def aplicar_filtro(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2
    # Adiciona padding de zeros para lidar com as bordas da imagem
    img_padded = np.pad(img, pad, mode='constant', constant_values=0)
    output = np.zeros_like(img)
    
    # Desliza o kernel sobre a imagem
    for i in range(h):
        for j in range(w):
            regiao = img_padded[i:i+kh, j:j+kw]
            valor = np.sum(regiao * kernel)
            output[i, j] = np.clip(valor, 0, 255) # Garante que o pixel fique entre 0-255
    return output

# Aplica Canny e Threshold para segmentar
def segmentar_imagem(img_gray):
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    bordas = cv2.Canny(img_gray, 100, 200)
    return thresh, bordas

# Ponto principal: Extração de features
def extrair_features_filtros(img):
    # 1. Pré-processamento: Padroniza o tamanho e cor
    img = cv2.resize(img, (100, 100))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Definição dos Kernels (Filtros)
    filtro_media = np.ones((3,3))/9
    filtro_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filtro_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filtro_laplaciano = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    filtro_sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    
    # 3. Aplicação dos filtros e segmentação
    media = aplicar_filtro(img_gray, filtro_media)
    sobel = aplicar_filtro(media, filtro_sobel_x) + aplicar_filtro(media, filtro_sobel_y)
    laplaciano = aplicar_filtro(media, filtro_laplaciano)
    sharpen = aplicar_filtro(media, filtro_sharpen)
    thresh, bordas = segmentar_imagem(img_gray)
    
    # 4. Extração de Features (Histogramas)
    # Decisão: Usar histogramas ao invés de média/std. É muito mais robusto.
    
    num_bins = 8  # 8 "grupos" de pixels
    hist_range = (0, 256) # Range de 0 a 255

    # density=True normaliza o histograma (importante para o SVM)
    # [0] pega apenas os valores do histograma
    hist_media = np.histogram(media, bins=num_bins, range=hist_range, density=True)[0]
    hist_sobel = np.histogram(sobel, bins=num_bins, range=hist_range, density=True)[0]
    hist_laplaciano = np.histogram(laplaciano, bins=num_bins, range=hist_range, density=True)[0]
    hist_sharpen = np.histogram(sharpen, bins=num_bins, range=hist_range, density=True)[0]
    hist_thresh = np.histogram(thresh, bins=num_bins, range=hist_range, density=True)[0]
    hist_bordas = np.histogram(bordas, bins=num_bins, range=hist_range, density=True)[0]

    # Junta tudo num vetorzão de features (6 filtros * 8 bins = 48 features)
    features = np.concatenate([
        hist_media,
        hist_sobel,
        hist_laplaciano,
        hist_sharpen,
        hist_thresh,
        hist_bordas
    ])
    
    return features

# Data Augmentation: Rotaciona e espelha para criar mais dados de treino
def gerar_augmentations(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgs_aug = [
        np.array(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)), # Original
        np.array(cv2.cvtColor(np.array(pil_img.rotate(90)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(pil_img.rotate(180)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(pil_img.rotate(270)), cv2.COLOR_RGB2BGR)),
        np.array(cv2.cvtColor(np.array(ImageOps.mirror(pil_img)), cv2.COLOR_RGB2BGR)), # Flip Horizontal
        np.array(cv2.cvtColor(np.array(ImageOps.flip(pil_img)), cv2.COLOR_RGB2BGR))    # Flip Vertical
    ]
    return imgs_aug

# -------------------- LÓGICA DE TREINAMENTO (ML) --------------------
def treinar_modelo():
    try:
        # Atualiza a GUI para o usuário
        label_status.configure(text="Treinando modelo... (aguarde)")
        app.update()
        
        features, labels = [], []
        
        # Calcula total de imagens para a barra de progresso
        total_imgs = sum([len([f for f in os.listdir(os.path.join(DATA_DIR,d)) if f.lower().endswith(('.jpg','.png','.jpeg'))])
                          for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
        
        if total_imgs == 0:
            messagebox.showerror("Erro", f"Nenhuma imagem encontrada em '{DATA_DIR}'.")
            label_status.configure(text="Erro: Nenhuma imagem encontrada.")
            return

        progress.configure(maximum=total_imgs, value=0)
        
        # Itera nas pastas (ex: 'latas' e 'outros')
        for class_name in os.listdir(DATA_DIR):
            class_path = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(class_path):
                # Rótulos: 0=lata, 1=outro
                label_value = 0 if class_name.lower() == 'latas' else 1
                
                # Itera em cada imagem
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg','.png','.jpeg')):
                        img_path = os.path.join(class_path, filename)
                        img = safe_imread(img_path)
                        
                        if img is not None:
                            # Aplica augmentation em cada imagem
                            for img_aug in gerar_augmentations(img):
                                # Extrai as features (histogramas)
                                feat = extrair_features_filtros(img_aug)
                                features.append(feat)
                                labels.append(label_value)
                            
                            # Atualiza a barra de progresso
                            progress.step()
                            app.update()
                            
        if len(features) < 10:
            messagebox.showerror("Erro", "Poucas amostras em 'data/latas' e 'data/outros'.")
            label_status.configure(text="Erro: poucas imagens.")
            return
            
        X = np.array(features)
        y = np.array(labels)
        
        # Normalização é CRUCIAL para o SVM
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Divisão 80% treino, 20% teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Kernel RBF. O 'linear' original era fraco demais para esses dados.
        model = SVC(kernel='rbf', C=1.0, random_state=42) 
        
        # Treina o modelo
        model.fit(X_train, y_train)
        
        # Avalia a acurácia no set de teste
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        
        # Salva o modelo E o scaler (importante salvar os dois)
        joblib.dump((model, scaler), MODEL_PATH)
        
        label_status.configure(text=f"Modelo treinado com sucesso! Acurácia: {acc:.2f}%")
        messagebox.showinfo("Sucesso", f"Modelo treinado!\nAcurácia: {acc:.2f}%")
        
    except Exception as e:
        messagebox.showerror("Erro no treinamento", str(e))
        label_status.configure(text="Erro ao treinar modelo.")
        print(f"[ERRO TREINAMENTO] {e}") # Log de erro no console

# Carrega o .pkl que contém o (modelo, scaler)
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Erro", "Modelo não encontrado. Treine primeiro!")
        return None, None
    model, scaler = joblib.load(MODEL_PATH)
    return model, scaler

# Classifica uma nova imagem
def classificar_imagem(img_path):
    model, scaler = carregar_modelo()
    if model is None:
        return
        
    img = safe_imread(img_path)
    if img is None: 
        label_status.configure(text="Erro: Não foi possível ler a imagem.")
        return
        
    # 1. Extrai features (MESMO PROCESSO DO TREINO)
    feat = extrair_features_filtros(img)
    
    # 2. Normaliza (SÓ .transform(), NUNCA .fit_transform() aqui)
    feat = scaler.transform([feat]) 
    
    # 3. Classifica
    pred = model.predict(feat)[0]
    resultado = "LATA" if pred == 0 else "OUTRO"
    
    # 4. Mostra na tela
    label_status.configure(text=f"Resultado: {resultado}")
    exibir_imagem(img_path, resultado)

# -------------------- FUNÇÕES DA INTERFACE (Callbacks) --------------------

# Callback do botão 'Selecionar Imagem'
def selecionar_imagem():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if file_path: # Se o usuário realmente escolheu um arquivo
        label_status.configure(text="Classificando imagem...")
        app.update()
        classificar_imagem(file_path)

# Mostra a imagem e o resultado na tela
def exibir_imagem(img_path, resultado_texto):
    try: 
        img = Image.open(img_path)
        img = img.resize((250, 250)) # Redimensiona para o canvas
        tk_img = ImageTk.PhotoImage(img)
        
        canvas_imagem.create_image(0, 0, anchor="nw", image=tk_img)
        canvas_imagem.image = tk_img # Guarda a referência (essencial no Tkinter)
        label_resultado.configure(text=resultado_texto)
    except Exception as e:
        label_status.configure(text=f"Erro ao exibir imagem: {e}")

# Callback do botão 'Ver Filtros'
def visualizar_filtros():
    # Esta função era do código original, não é mais tão útil
    # já que não salvamos os filtros, mas mantida.
    arquivos = [f for f in os.listdir(RESULTADOS_DIR) if f.endswith(".jpg")]
    if not arquivos:
        messagebox.showinfo("Aviso", "Nenhum filtro salvo ainda.")
        return
    for f in arquivos:
        img = cv2.imread(os.path.join(RESULTADOS_DIR, f))
        cv2.imshow(f, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Roda o treino numa thread separada
def thread_treinar():
    # Isso evita que a interface gráfica (GUI) trave durante o treino
    t = threading.Thread(target=treinar_modelo)
    t.start()

# -----------------------------------------------------------------
# CONSTRUÇÃO DA INTERFACE (TTKBOOTSTRAP)
# -----------------------------------------------------------------

# Janela principal
app = tb.Window(themename="minty")
app.title("EcoCan - Tecnologia Sustentável")
app.geometry("650x700") # Tamanho fixo
app.resizable(False, False)

# --- Logo EcoCan ---
try:
    # Tenta carregar a imagem do logo
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
    # Se falhar, usa um texto como fallback
    print(f"[AVISO] Falha ao carregar logo: {e}")
    lbl_logo = tb.Label(app, text="EcoCan", font=("Segoe UI", 26, "bold"), bootstyle="success")
    lbl_logo.pack(pady=(25, 10))

# --- Título ---
lbl_titulo = tb.Label(app, text="Tecnologia em prol da Reciclagem Sustentável",
                      font=("Segoe UI", 15, "bold"))
lbl_titulo.pack(pady=(0, 20))

# --- Botões (Estilizados) ---
frame_btns = tb.Frame(app)
frame_btns.pack(pady=9)

verde_principal = "#56cc9d"
verde_hover = "#D9F1DA"

# Função para aplicar estilo customizado e hover
def estilizar_botao(btn):
    btn.configure(
        bootstyle="success",
        width=15,
        cursor="hand2", # Cursor de "mãozinha"
        style="Custom.TButton"
    )
    # Efeito de hover (mouse em cima)
    btn.bind("<Enter>", lambda e: btn.configure(style="Hover.TButton"))
    # Efeito de hover (mouse sai)
    btn.bind("<Leave>", lambda e: btn.configure(style="Custom.TButton"))

# Define os estilos (similar a CSS)
style = tb.Style()
style.configure("Custom.TButton", background=verde_principal, foreground="white",
                font=("Segoe UI", 10, "bold"), borderwidth=0, relief="flat",
                focusthickness=3, focuscolor=verde_principal, padding=10, border=0)
style.map("Custom.TButton", relief=[("pressed", "flat")])
style.configure("Hover.TButton", background=verde_hover, foreground="white",
                font=("Segoe UI", 10, "bold"), padding=10)

# Cria os botões
btn_treinar = tb.Button(frame_btns, text="Treinar Modelo", command=thread_treinar)
btn_classificar = tb.Button(frame_btns, text="Selecionar Imagem", command=selecionar_imagem)
btn_filtros = tb.Button(frame_btns, text="Ver Filtros", command=visualizar_filtros)

# Posiciona os botões lado a lado (em grid)
for i, btn in enumerate([btn_treinar, btn_classificar, btn_filtros]):
    btn.grid(row=0, column=i, padx=10)
    estilizar_botao(btn)

# --- Display da Imagem ---
frame_img = tb.Frame(app, bootstyle="success", padding=8) # Moldura verde
frame_img.pack(pady=20)
canvas_imagem = tb.Canvas(frame_img, width=250, height=250, background="#C8E6C9", highlightthickness=0)
canvas_imagem.pack()

# --- Labels de Status ---
label_resultado = tb.Label(app, text="", font=("Segoe UI", 14, "bold")) # "LATA" ou "OUTRO"
label_resultado.pack(pady=(10, 5))
label_status = tb.Label(app, text="Aguardando ação...", font=("Segoe UI", 10)) # "Treinando...", "Aguardando..."
label_status.pack(pady=(5, 10))

# --- Barra de Progresso ---
progress = tb.Progressbar(app, orient="horizontal", length=400, mode="determinate", bootstyle=SUCCESS)
progress.pack(pady=(5, 20))

# --- Loop Principal ---
app.mainloop() # Inicia a aplicação e espera pela interação do usuário