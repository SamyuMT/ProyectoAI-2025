import requests
from typing import Optional, Dict
import pandas as pd
import json
import joblib
import fitz  # PyMuPDF

# Simple in-memory cache
script_cache = {}

JSON_TEMPLATE = {
    "DPTO_MPIO": 0, "ANO_CENSO": 0, "TRIMESTRE": 0, "ANIO_MESINICIO": 0,
    "DESTINO2": 0, "ESTADO_ACT": 0, "ESTRATO": 0, "LOCALIDAD2": 0, "AMPLIACION": 0, "OB_FORMAL": 0, "USO": 0, "USO_DOS": 0,
    "AREATOTCO2": 0, "AREATOTZC": 0, "AREAVENUNI": 0, "AREAVENDIB": 0, "AREA_NOVIS": 0,
    "AREA_RANVIVI_1": 0, "AREA_RANVIVI_2": 0, "AREA_RANVIVI_3": 0, "AREA_RANVIVI_4": 0, "AREA_RANVIVI_5": 0, "AREA_RANVIVI_6": 0,
    "UNI_DEC_NOVIS": 0, "UNIDEC_RANVIVI_1": 0, "UNIDEC_RANVIVI_2": 0, "UNIDEC_RANVIVI_3": 0, "UNIDEC_RANVIVI_4": 0,
    "UNIDEC_RANVIVI_5": 0, "UNIDEC_RANVIVI_6": 0, "NUMUNIDEST": 0, "NUMUNIVEN": 0, "NUMUNIXVEN": 0,
    "NRO_EDIFIC": 0, "NRO_PISOS": 0, "MANO_OBRAP": 0, "MANO_OBRAT": 0, "MANO_OBRAF": 0,
    "PRECIOVTAX": 0, "TIPOVRDEST": 0, "CAPITULO": 0, "GRADOAVANC": 0,
    "C1_EXCAVACION": 0, "C1_CIMENTACION": 0, "C1_DESAGUES": 0, "C2_ESTRUCTURA": 0, "C2_INST_HIDELEC": 0, "C2_CUBIERTA": 0,
    "C3_MAMPOSTERIA": 0, "C3_PANETE": 0, "C4_PISO_ENCHAPE": 0, "C4_CARP_METALICA": 0, "C4_CARP_MADERA": 0,
    "C4_CIELO_RASO": 0, "C5_VID_CERRAJERIA": 0, "C5_PINTURA": 0, "C6_REM_EXTERIORES": 0, "C6_REM_ACABADOS": 0, "C6_ASEO": 0,
    "SIS_CONSTR": 0, "LIC_RADICADO_SN": 0, "MOVIMIENTO_ENC": 0, "RANVIVI": 0
}

def build_prompt(text_pdf, json_template=JSON_TEMPLATE):
    return f"""
EXTRACCIÓN DE FORMULARIO CEED DANE

INSTRUCCIONES:
1. Extrae del texto siguiente SOLO los valores para las siguientes variables (si no están, pon 0).
2. Devuelve SOLO el JSON con los valores encontrados, siguiendo exactamente la estructura y el orden del JSON de ejemplo.
3. No devuelvas texto adicional, ni comentarios.
4. Si no encuentras un valor, pon 0 en su lugar.
5. Asegúrate de que el JSON sea válido y esté bien formateado. no agregues ```json al inicio ni al final.
6. El JSON debe comenzar directamente con {{ y terminar con }}

Texto del formulario:
{text_pdf}

JSON de ejemplo (solo debes cambiar los valores por los del formulario):
{json.dumps(json_template, indent=2)}
"""

def extract_text_from_pdf(pdf_source):
    try:
        # Handle both file paths (str) and file objects
        if isinstance(pdf_source, str):
            # Case 1: pdf_source is a file path
            doc = fitz.open(pdf_source)
        else:
            # Case 2: pdf_source is a file object
            pdf_data = pdf_source.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None
    

class AgentEstimacionCEED:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })

    def _call_ai(self, prompt):
        try:
            response = self.session.post(
                "https://api.deepseek.com/chat/completions",
                json={
                    "model": "deepseek-chat",  # Puedes cambiar el modelo aquí, por ejemplo: "deepseek-coder", "deepseek-llm", etc.
                    "messages": [{
                        "role": "system",
                        "content": "Eres un extractor de datos especializado en formularios CEED del DANE. Responde EXCLUSIVAMENTE con un JSON válido."
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    "temperature": 0.3
                }
            )
            if response.status_code != 200:
                raise Exception(f"Error de API: {response.status_code} - {response.text}")
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling AI API: {str(e)}")
            return "Error generando el guión. Por favor intenta de nuevo."

    def process_ceed_form(self, pdf_source: str) -> Optional[Dict]:
        # Paso 1: Extraer texto del PDF
        text_pdf = extract_text_from_pdf(pdf_source)
        if not text_pdf:
            return None
            
        # Paso 2: Generar el prompt
        prompt = build_prompt(text_pdf)
        # Paso 3: Llamar a la IA
        return self._call_ai(prompt)
    

    def mensaje_lindo(self, text) -> Optional[Dict]:
           
        # Paso 2: Generar el prompt
        prompt = f"""
    Redacta un mensaje breve, amigable y claro para informar al usuario el precio estimado del metro cuadrado del terreno: {text} pesos colombianos por metro cuadrado. 
    El mensaje debe ser cordial, incluir emojis, resaltar el valor y transmitir entusiasmo. Evita frases largas o tecnicismos. 
    Ejemplo: "¡Excelente noticia! El precio estimado por metro cuadrado es de {text} pesos😊
    usa emojis para hacerlo más atractivo y amigable.
    
    tener en cuenta que si el numero no debe ir entre comillas, ni comas, ni [], tienes que tener $ valor COP"
    """
        response = self._call_ai(prompt)
        try:
            response_json = json.loads(response.replace('\n', '').replace('```json', '').replace('```', ''))
            mensaje = response_json.get("mensaje", "")
        except Exception:
            mensaje = response
        return mensaje

