from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import time
from collections import defaultdict
from flask_cors import CORS


request_log = defaultdict(list)
RATE_LIMIT = 10
RATE_PERIOD = 600  # 10 minutes

load_dotenv()
print("🔑 Loaded OpenAI key:", os.getenv("OPENAI_API_KEY")[:10], "..." if os.getenv("OPENAI_API_KEY") else "❌ NOT FOUND")

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """
You are a professional visual prompt engineer.

You generate short, visually precise prompts for creating a photo-realistic background image. A product image will be composited into this background later using a masked overlay. Your task is to describe only the background setting.

The product is not present in the generation process. Instead, the image must be constructed in such a way that a product can be placed into the center afterward, with seamless integration in terms of lighting, perspective, and realism.

Strict instructions:
- DO NOT describe or include the product.
- DO NOT place any object, furniture, person, or shadow in the center of the image.
- DO NOT include people, hands, marketing language, or storytelling.
- DO NOT include placeholder elements or central focal points.
- DO NOT mention the product name.

Focus entirely on:
- Creating a realistic environment (e.g. office, living room, physiotherapy space).
- Matching the selected camera angle (front, side, or close-up).
- Lighting and shadows that make the product blend naturally when added.
- Walls, floors, decor elements, and background texture.
- Ensuring the center of the image is visually clean and empty.
- The image must look like a square (2000x2000px) camera photograph.

The final image will be used as a background only. The product will be masked and added later. Your prompt defines the scene into which the product must seamlessly fit.
"""


@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    ip = request.remote_addr
    now = time.time()
    # Filter requests within time window
    request_log[ip] = [t for t in request_log[ip] if now - t < RATE_PERIOD]
    if len(request_log[ip]) >= RATE_LIMIT:
        return jsonify({"error": "Rate limit exceeded. Please wait before trying again."}), 429
    request_log[ip].append(now)

    data = request.get_json()
    print("📦 Inkomende data:", data)
    print("Selected Template:", data.get("selectedTemplate"))  # Log the selected template
    template_instruction = ""
    if data.get("selectedTemplate") == "Bejaardenhuis":
        template_info = """
        Genereer een warme, uitnodigende woonkamer in een bejaardenhuis, met lichtgrijze muren, houten vloer in licht eiken, en zachte daglichtinval van links. De ruimte is ruim, realistisch, opgeruimd en bevat eventueel subtiele elementen zoals een fauteuil of bijzettafel aan de rand van het beeld. Het midden van de afbeelding moet volledig leeg blijven — geen meubels, objecten of schaduwen. Alleen achtergrond en sfeer. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Sportschool":
        template_info = """
        Genereer een moderne, realistische fitnessruimte met rubberen vloer, matgrijze muren en zachte verlichting. Grote spiegels en metalen details zijn toegestaan aan de randen. Geen mensen of apparaten in het midden. Laat de vloer in het midden leeg en visueel toegankelijk, zonder obstakels. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Kantoor":
        template_info = """
        Toon een moderne, lichte kantoorruimte in Scandinavische stijl met off-white muren, visgraatvloer in licht hout en rustige inrichting. Links valt natuurlijk licht binnen via een groot raam met transparante gordijnen. Het midden van het beeld blijft leeg voor productcompositing. Gebruik zachte decor aan de zijkanten zoals planten of een bureau. Geen mensen of objecten in het midden. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Fysiotherapie":
        template_info = """
        Genereer een cleane, moderne fysiotherapieruimte met behandelbanken aan de zijkant, anatomieposters aan de muur, en pastelkleuren op neutrale wanden. Zorg voor een steriele sfeer met zachte blur en diffuus daglicht. Het midden blijft volledig leeg. Geen mensen of apparatuur. Alleen achtergrond. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Straat":
        template_info = """
        Toon een rustige Nederlandse straat met realistische bestrating, groene struiken, lantaarnpaal of fietsen aan de rand. Geen auto's of personen. Laat het midden van het beeld leeg en open. De compositie moet licht en realistisch zijn, met zachte schaduwen. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Woonkamer":
        template_info = """
        Genereer een ruime, moderne woonkamer in Scandinavische stijl met een licht houten vloer, witte of beige muren, en zacht daglicht van links. De sfeer is rustig, warm en minimalistisch. Gebruik decoratie zoals een plant of fauteuil uitsluitend aan de randen van het beeld. Laat het midden volledig leeg voor productcompositing. Geen objecten of schaduw in het midden. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Hal":
        template_info = """
        Toon een ruime, opgeruimde hal met lichtgrijze muren, houten vloer in Scandinavische stijl, en zacht daglicht dat van links binnenvalt. Geen jassen, schoenen of accessoires in beeld. Het midden van de afbeelding moet leeg blijven. Alleen achtergrond. Resolutie: 2000 x 2000 pixels.
        """
    elif data.get("selectedTemplate") == "Badkamer":
        template_info = """
        Genereer een moderne inloopdouche met beige keramische tegels, warme zandtinten, en natuurlijk licht van links. Geen flessen of accessoires. Laat het midden leeg voor plaatsing van een product. Alleen achtergrondtextuur, geen objecten. Resolutie: 2000 x 2000 pixels.
        """
    else:
        template_info = "Standaardafbeelding van het product in een neutrale, fotorealistische setting."


    user_prompt = f"""
Je taak is om een fotorealistische achtergrondafbeelding te genereren waarin later een product (de {data.get("productnaam", " ")}) geplakt zal worden. Gebruik het volgende camerastandpunt: {data.get("view", "")} zicht.

De achtergrond moet volledig realistisch en natuurlijk ogen, in lijn met de volgende omgeving:

{template_info}

⚠️ Belangrijke eisen:
- Het midden van de afbeelding moet volledig leeg blijven: geen meubels, geen objecten, geen decoratie, geen schaduwen.
- De belichting en het perspectief moeten zo gekozen zijn dat het geplakte product later natuurlijk integreert.
- De afbeelding moet vierkant zijn: 2000 x 2000 pixels.
- Geen enkel product of mens mag aanwezig zijn.

Genereer uitsluitend een achtergrond. Het product wordt later toegevoegd via compositing.
"""

    print("📨 Gebruikersprompt:")
    print(user_prompt)

    if data.get("selectedTemplate") != "AI_generated":
        return jsonify({"prompt": user_prompt.strip()})


# Nieuwe route voor het genereren van een afbeelding via prompt + upload
@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt is vereist"}), 400

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return jsonify({"image_url": response.data[0].url})
    except Exception as e:
        print("❌ Backend error bij /generate-image:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)