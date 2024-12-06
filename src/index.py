import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from xml.dom import minidom #formatar o xml com identação correta

# Classe que define os intervalos das cores e detecta essas cores em partes das imagens (quadrantes)
class EloMalucoCores:
    @staticmethod
    def definir_intervalos_cores():
        return {
            # O intervalo das cores foi definido usando o HSV, por isso são dois intervalos para definir o vermelho. Já que o vermelho esta representado no inicio e no final da escala do HSV.
            "vermelho1": (np.array([0, 120, 70]), np.array([10, 255, 255])),
            "vermelho2": (np.array([170, 120, 70]), np.array([180, 255, 255])),
            "verde": (np.array([35, 100, 100]), np.array([85, 255, 255])),
            "branco": (np.array([0, 0, 200]), np.array([180, 30, 255])),
            "cinza": (np.array([0, 0, 50]), np.array([180, 50, 200])),
            "amarelo": (np.array([30, 100, 100]), np.array([50, 255, 255]))
        }
    # Detecta a quantidade de pixels de uma cor em um quadrante
    @staticmethod
    def detectar_cor(quadrante, intervalo_min, intervalo_max):
        mascara = cv2.inRange(quadrante, intervalo_min, intervalo_max)# filtra a cor
        return cv2.countNonZero(mascara) # retorna a contagem de pixels

    # Função para detectar a cor que mais aparece em cada uma das partes do cubo (quadrnates)
    @classmethod
    def detectar_cores_quadrantes(cls, imagem):
        imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV) #inclue a conversão para HSV
        intervalos = cls.definir_intervalos_cores()
        altura, _, _ = imagem.shape

        #Dividimos a imagem em 4 partes diferentes (1/4 da altura para cada)
        quadrantes = [
            imagem_hsv[0:altura//4, :],               
            imagem_hsv[altura//4:altura//2, :],     
            imagem_hsv[altura//2:3*altura//4, :],    
            imagem_hsv[3*altura//4:, :]             
        ]

        cores_predominantes = []
        # busca em cada parte a cor que mais aparece
        for quadrante in quadrantes:
            cores = {
                "vermelho": cls.detectar_cor(quadrante, intervalos["vermelho1"][0], intervalos["vermelho1"][1]) +
                            cls.detectar_cor(quadrante, intervalos["vermelho2"][0], intervalos["vermelho2"][1]),
                "verde": cls.detectar_cor(quadrante, intervalos["verde"][0], intervalos["verde"][1]),
                "branco": cls.detectar_cor(quadrante, intervalos["branco"][0], intervalos["branco"][1]),
                "cinza": cls.detectar_cor(quadrante, intervalos["cinza"][0], intervalos["cinza"][1]),
                "amarelo": cls.detectar_cor(quadrante, intervalos["amarelo"][0], intervalos["amarelo"][1])
            }
            cores_predominantes.append(max(cores, key=cores.get))

        return cores_predominantes #retorna a cor que mais aparece

# Identificar qual padrão de textura aparece na imagem
class EloMalucoTextura:
    @staticmethod
    #Busca na parte se ele tem uma das cores do elo
    def verificar_cor(parte, intervalos_cores):
        imagem_hsv = cv2.cvtColor(parte, cv2.COLOR_BGR2HSV)
        cores_detectadas = {
            "vermelho": cv2.countNonZero(cv2.inRange(imagem_hsv, intervalos_cores["vermelho1"][0], intervalos_cores["vermelho1"][1])) + 
                        cv2.countNonZero(cv2.inRange(imagem_hsv, intervalos_cores["vermelho2"][0], intervalos_cores["vermelho2"][1])),
            "verde": cv2.countNonZero(cv2.inRange(imagem_hsv, intervalos_cores["verde"][0], intervalos_cores["verde"][1])),
            "branco": cv2.countNonZero(cv2.inRange(imagem_hsv, intervalos_cores["branco"][0], intervalos_cores["branco"][1])),
            "amarelo": cv2.countNonZero(cv2.inRange(imagem_hsv, intervalos_cores["amarelo"][0], intervalos_cores["amarelo"][1]))
        }
        return any(cores_detectadas.values()) # Retorna true se aquela parte tem uma dessas cores

    # Analisa cada par de parte, que pertence a uma peça do elo
    @classmethod
    def analisar_pares(cls, partes, cores_predominantes, intervalos_cores):

        resultados_pares = []

        for i in range(0, 8, 2):

            # As partes de uma mesma peça são consecutivas:
            parte1 = partes[i]
            parte2 = partes[i + 1]

            # usa a função para verificar se essas partes são coloridas ou não
            cor_parte1 = cls.verificar_cor(parte1, intervalos_cores)
            cor_parte2 = cls.verificar_cor(parte2, intervalos_cores)

            # Verifica em qual dos elementos dos pares existe cor, e define a qual textura a peça pertence
            if cor_parte1 and not cor_parte2:
                resultado = "superior"
            elif cor_parte1 and cor_parte2:
                resultado = "meio"
            elif not cor_parte1 and cor_parte2:
                resultado = "inferior"
            else:
                resultado = "vazio"

            resultados_pares.append((cores_predominantes[i//2], resultado))

        return cls.mapear_codigos(resultados_pares) # mapeamento para achar qual a correspondente para o resultado

    @staticmethod
    #Aqui é realizado o mapeamento
    def mapear_codigos(resultados_pares):
        cores_map = {
            'vazio': 'vzo', 
            'amarelo': 'am', 
            'vermelho': 'vm', 
            'branco': 'br', 
            'verde': 'vr', 'cinza': 'vzo'
        }
        posicao_map = {
            'superior': 's', 
            'meio': 'm', 
            'inferior': 'i', 
            'vazio': ''
        }
        
        return [cores_map.get(cor, 'vzo') + posicao_map[posicao] for cor, posicao in resultados_pares]

# Classe para salvar os dados mapeados em um xml
class EloMalucoXML:
    @staticmethod
    def salvar_resultados_xml(resultados_matriz, caminho_base='output/output.xml'):
        nome_base, extensao = os.path.splitext(caminho_base) #para não sobreescrever os arquivos, divide o caminho base
        contador = 1
        caminho_arquivo = caminho_base

        # Garente que ele não sobreescreve um existente
        while os.path.exists(caminho_arquivo):
            caminho_arquivo = f"{nome_base}{contador}{extensao}"
            contador += 1

        root = ET.Element("EloMaluco") # cria a raiz do xml

        estado_atual = ET.SubElement(root, "EstadoAtual")
        
        # para cada linha dos resultados adiciona uma nova linha do xml
        for linha in resultados_matriz:
            row = ET.SubElement(estado_atual, "row")
            
            for codigo in linha:
                ET.SubElement(row, "col").text = codigo
        
        xml_str = ET.tostring(root, encoding='utf-8')
        
        # Garante a identação
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="    ")  
        
        # Salva
        with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
            arquivo.write(pretty_xml)
        
        print(f"Resultados salvos em {caminho_arquivo}")

# Class do processamento de imagens
class EloMalucoDetector:
    def __init__(self):

        #Para armazenar a matriz de resultados e as cores que mais aparecem no quadrante
        self.resultados_matriz = [] 
        self.cores_predominantes = []

    # Função que detecta as bordas da imagem e realiza o corte na borda
    def detectar_bordas(self, imagem_caminho, ponto_destino, altura_desejada=800, largura_corte=60):

        imagem = cv2.imread(imagem_caminho) # le a imagem

        # Calcula a largura para fazer o redimensionamento da imafgen
        largura_desejada = int(altura_desejada * imagem.shape[1] / imagem.shape[0])

        tamanho_final = (largura_desejada, altura_desejada)

        # converte para a escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # suaviza a imagens para melhorar a detecção de bordas
        imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

        # Usa o metodo Canny para detectar as bordas
        bordas = cv2.Canny(imagem_suavizada, 50, 150)

        # Acha os contornos dessa imagem
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            print("Nenhum contorno encontrado!")
            return bordas

        # pega o contorno de fora como principal
        maior_contorno = max(contornos, key=cv2.contourArea)
        
        # Pega esse contorno como base para marcar um retangulo que define o tamanho da imagem
        x, y, w, h = cv2.boundingRect(maior_contorno)

        # Corta a imagem com base no contorno
        imagem_cortada = imagem[y:y+h, x:x+w]
        
        # Redimenciona essa imagem cortada
        imagem_redimensionada = cv2.resize(imagem_cortada, tamanho_final)

        altura, largura = imagem_redimensionada.shape[:2]

        largura_corte = min(largura_corte, largura)

        # Define o centro da imagem para o corte central
        centro_x = largura // 2

        left = centro_x - largura_corte // 2
        right = centro_x + largura_corte // 2

        imagem_corte_central = imagem_redimensionada[:, left:right]

        # Divide as oito partes para fazer a busca de cor
        partes = []
        altura, largura = imagem_corte_central.shape[:2]

        altura_parte = altura // 8  

        for i in range(8):
            parte = imagem_corte_central[i * altura_parte:(i + 1) * altura_parte, :]
            partes.append(parte)

        # Obtem os intervalos de cores
        intervalos_cores = EloMalucoCores.definir_intervalos_cores()

        # Detecta as cores prodominantes
        self.cores_predominantes = EloMalucoCores.detectar_cores_quadrantes(imagem_redimensionada)

        # Chama a analise de partes dos pares
        resultados_imagem = EloMalucoTextura.analisar_pares(partes, self.cores_predominantes, intervalos_cores)
        
        # Armazena esse resultado na matriz
        self.resultados_matriz.append(resultados_imagem)

        return partes

def main():

    ponto_destino = (200, 250)
    
    caminho_imagens = [
        'data/Ex_input01_01.png',
        'data/Ex_input01_02.png',
        'data/Ex_input01_03.png',
        'data/Ex_input01_04.png'
    ]

    detector = EloMalucoDetector()

    # Processa cada imagem dentrro do array
    for imagem in caminho_imagens:
        print(f"Processando {imagem}...")
        detector.detectar_bordas(imagem, ponto_destino)

    print("\nMatriz de Resultados:")
    for linha in detector.resultados_matriz:
        print(linha)

    # Salva no xml
    EloMalucoXML.salvar_resultados_xml(detector.resultados_matriz, 'output/output.xml')

if __name__ == "__main__":
    main()