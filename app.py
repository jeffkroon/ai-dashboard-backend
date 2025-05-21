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
        Toon een looprek in een ruime, veilige woonkameromgeving die is ingericht voor ouderen. De ruimte heeft warme houten vloeren, een lichtgrijze of beige muur, en is ingericht met comfortabele, klassieke meubels zoals een zachte fauteuil, een houten bijzettafel met een leeslamp, en een vloerkleed met antislip. Laat natuurlijk daglicht binnenvallen via een groot raam met transparante gordijnen. Het looprek staat iets links van het midden, volledig zichtbaar, met ongeveer 2/4 van de ruimte rechts leeg voor tekst of USP’s. De compositie is warm, huiselijk en functioneel, met kleine details zoals een plant op de vensterbank of een leesbril op tafel. Geen visuele afleiding zoals snoeren of rommel. Resolutie: 2000 x 2000 pixels.

        Photo-realistic living room environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Sportschool":
        template_info = """
        Photo-realistische afbeelding van het product gepositioneerd in een moderne, hoogwaardige gymruimte. De omgeving bevat gymapparatuur, rubberen vloer, grote spiegels en industriële verlichting. Het product staat centraal en is duidelijk zichtbaar. De scène voelt schoon, energiek en gefocust aan. Gebruik filmische belichting en scherpe focus om enkele USP’s te accentueren. Geen mensen — alleen sfeer en productpresentatie. Resolutie: 2000 x 2000 pixels.

        Photo-realistic gym environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Kantoor":
        template_info = """
        Toon een ruime, moderne kantoorhoek (voor deze bureaustoel) in Scandinavische stijl met een serene, opgeruimde uitstraling. De ruimte is afgewerkt met mat afgewerkte, off-white muren en een visgraatvloer van licht eikenhout. Links valt zacht natuurlijk daglicht binnen via een groot raam met transparante, vloerlange gordijnen. Rechts staat een minimalistisch houten bureau met een opengeklapte laptop, een zwart koffiekopje, een gesloten notitieboekje en een kleine kamerplant in een keramische pot. De achterwand bevat een zwart stalen glazen scheidingswand, die de werkruimte subtiel spiegelt en extra ruimtelijkheid toevoegt. Eén wandcontactdoos is zichtbaar op de muur. De compositie is hyperrealistisch, licht en professioneel, met voldoende open vloeroppervlak voor een stoel. Er is geen visuele afleiding zoals losse kabels, papier of persoonlijke rommel. Resolutie: 2000 x 2000 pixels.

        Photo-realistic office environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Fysiotherapie":
        template_info = """
        Photo-realistische close-up van het product geplaatst in een fysiotherapieruimte. De achtergrond bevat behandelbanken, anatomieposters en zachte pastelkleuren op de muren. Focus op materiaaldetails, precisieontwerp en medische toepasbaarheid. Benadruk enkele USP’s via heldere belichting en macro-detail. Zorg voor een steriele sfeer met zachte blur en natuurlijk licht — geen mensen. Resolutie: 2000 x 2000 pixels.

        Photo-realistic physiotherapy room environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Straat":
        template_info = """
        Cinematische, photo-realistische afbeelding van het product geplaatst in een stedelijke of suburbane straatomgeving. Realistische bestrating, zachte schaduwen, Nederlandse elementen zoals fietsen, groen, gebouwen of borden. Het product springt eruit maar voelt geïntegreerd in de scène. Gebruik dynamisch natuurlijk licht (zoals golden hour of zacht bewolkt). Leg nadruk op lifestyle en toepasbaarheid. Geen mensen — alleen context en sfeer. Resolutie: 2000 x 2000 pixels.

        Photo-realistic street environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Woonkamer":
        template_info = """
        Toon deze rollator in een ruime, realistische woonkameromgeving met natuurlijke belichting en een rustige, neutrale achtergrond. Zorg dat het product goed zichtbaar is, met 2/4 lege ruimte aan de rechterzijde van de afbeelding voor het toevoegen van USP’s of tekst. De compositie moet warm en licht, functioneel en uitnodigend zijn, met eventueel subtiele decoratieve elementen zoals planten, een kleed of meubilair op afstand — zonder visuele afleiding naast het product. Extra realistische look. Resolutie: 2000 x 2000 pixels.

        Photo-realistic living room environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Hal":
        template_info = """
        Toon (Solelution achillespees zooltjes) in een ruime, moderne hal van een huis bij de kapstok. De hal heeft een realistische afmeting en is afgewerkt met lichte muren en een houten vloer, en een rustige, opgeruimde indeling (bijvoorbeeld Scandinavisch). Laat natuurlijk daglicht zacht invallen van links om een warme, functionele sfeer te creëren. Plaats het product iets links van het midden op de vloer, volledig zichtbaar. Aan de rechterzijde blijft ongeveer 2/4 van de afbeelding leeg, bedoeld voor het toevoegen van USP’s of tekst. Zet een paar schoenen netjes ernaast om context te geven. De compositie is hyperrealistisch, warm en licht, zonder visuele afleiding zoals trap, jassen of accessoires. Resolutie: 2000 x 2000 pixels.

        Photo-realistic hallway environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    elif data.get("selectedTemplate") == "Badkamer":
        template_info = """
        Toon (douchekruk) in een ruime, moderne inloopdouche. De ruimte is betegeld met warme, beige keramische wand- en vloertegels met een lichte zandtint. Laat natuurlijk daglicht zacht invallen van links om een uitnodigende, functionele sfeer te creëren. Plaats het product iets links van het midden, volledig zichtbaar. Laat aan de rechterzijde ongeveer 2/4 van de afbeelding leeg voor het toevoegen van tekst of USP’s. Zet één realistisch object zoals een bruine fles douchegel op de vloer. Voeg in het midden van de vloer een subtiel, verzonken doucheputje toe van geborsteld RVS met diepte, schaduwwerking en tegelverloop naar het afvoerputje. De compositie is hyperrealistisch, warm en licht, zonder visuele afleiding zoals planten of accessoires. Resolutie: 2000 x 2000 pixels.

        Photo-realistic bathroom environment.
        The image must not contain any product, object, furniture, person, or focal subject in the center.
        The center must remain completely empty — no shadows, no placeholder.
        Only background elements such as walls, floor, textures, and ambient lighting. This image is for background compositing only.
        """
    else:
        template_info = "Standaardafbeelding van het product in een neutrale, fotorealistische setting."


    user_prompt = f"""
    
    
{template_instruction}
Maak een afbeelding van de ({data.get("productnaam", " ")}) vanuit een  {data.get("view", "")} zicht.
{template_info}

"""
    print("📨 Gebruikersprompt:")
    print(user_prompt)

    if data.get("selectedTemplate") != "AI_generated":
        return jsonify({"prompt": user_prompt.strip()+ "\n \n Genereer dit beeld in 2000 X 2000 pixels \n \n  !IMPORTANT: Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?"})
    try:
        user_prompt = f"""
Focus on the product's key features: color, texture, lighting, and perspective. Avoid any unnecessary details, explanations, or storytelling.

Ensure that the product remains exactly the same as the original — no changes to shape, size, color, or material. The {data.get("productnaam", " ")} must stay consistent across all images.

Take into account the selected environment: {template_info, ""}, including background elements, textures, and lighting direction. The background must match the required consistency for the setting.

Include any extra descriptions provided: {data.get("extraDescription", "")}.
"""

        completion = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9,
        max_tokens=400)
        result = completion.choices[0].message.content.strip() + "\n \n   !IMPORTANT Genereer eerst voor jezelf een afbeelding met product en stuur mij de afbeelding zonder product erin kan dat?"
        return jsonify({"prompt": result})
    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": str(e)}), 500



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