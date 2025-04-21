#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import base64
import io
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ——— Configuration Logging —————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ——— Modèles globaux —————————————————————————————————————
URBAN_DENSITY_MODEL = None
ACCESSIBILITY_MODEL = None
OBJECT_MODEL = None  # modèle pour /identify

def initialize_models():
    global URBAN_DENSITY_MODEL, ACCESSIBILITY_MODEL, OBJECT_MODEL

    # Modèles urbains
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 10, 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    URBAN_DENSITY_MODEL = RandomForestClassifier(n_estimators=20, random_state=42)
    URBAN_DENSITY_MODEL.fit(X_train, y_train)
    ACCESSIBILITY_MODEL = RandomForestClassifier(n_estimators=20, random_state=42)
    ACCESSIBILITY_MODEL.fit(X_train, y_train)

    # Modèle dummy pour identify (on simule ici un RandomForest)
    X_obj = np.random.rand(100, 10)  # 10 features extraites d'image
    y_obj = np.random.randint(0, 5, 100)  # 5 classes d'objets
    OBJECT_MODEL = RandomForestClassifier(n_estimators=15, random_state=42)
    OBJECT_MODEL.fit(X_obj, y_obj)

    logger.info("✅ Modèles d'IA initialisés")

# ——— App & CORS —————————————————————————————————————————
def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    initialize_models()
    return app

app = create_app()

# ——— Fonctions utilitaires ——————————————————————————————
def calculate_area(coords):
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def decode_image(base64_str):
    """Retourne un objet PIL.Image à partir d'un DataURL base64."""
    header, b64 = base64_str.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def extract_image_features(img: Image.Image):
    """
    Simule l'extraction de features d'une image PIL.
    Ici on redimensionne et on prend les moyennes de canaux.
    """
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    # 10 features : moyennes et écart-types sur R, G, B
    feats = []
    for c in range(3):
        feats.append(arr[..., c].mean())
        feats.append(arr[..., c].std())
    feats.append(arr[..., :2].mean())  # 9
    feats.append(arr[..., 1:].mean())  # 10
    return np.array(feats).reshape(1, -1)

# ——— Endpoints ——————————————————————————————————————————

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Existant : reçoit JSON { location, buildings, roads, water, landUse }
    Retourne patterns, recommandations, stats, heatmap.
    """
    data = request.get_json(force=True)
    try:
        loc       = data["location"]
        buildings = data.get("buildings", [])
        roads     = data.get("roads", [])
        water     = data.get("water", [])
        landuse   = data.get("landUse", [])
        # -- validation simple
        assert isinstance(loc, list) and len(loc) == 2
    except Exception as e:
        logger.error("❌ Analyse payload invalide: %s", e)
        return jsonify({"error": "Payload invalide"}), 400

    # Stats de base
    stats = {
        "building_count": len(buildings),
        "road_count":     len(roads),
        "water_count":    len(water),
        "landuse_count":  len(landuse),
    }
    # Aire totale
    stats["total_building_area_m2"] = round(
        sum(calculate_area(b.get("coords", [])) for b in buildings), 2
    )
    # Scores IA simulés
    stats["urban_density_score"]   = float(URBAN_DENSITY_MODEL.predict([[stats["building_count"],0,0,0,0]])[0])
    stats["accessibility_score"]   = float(ACCESSIBILITY_MODEL.predict([[stats["road_count"],0,0,0,0]])[0])

    # Patterns & recommandations
    patterns = identify_urban_patterns(buildings, roads, water, landuse)
    recommendations = generate_recommendations(buildings, roads, water, landuse, patterns)

    # Heatmap simulée (vide ici pour alléger)
    heatmap = []

    return jsonify({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "location": loc,
        "stats": stats,
        "patterns": patterns,
        "recommendations": recommendations,
        "density_heatmap": heatmap
    }), 200

@app.route("/identify", methods=["POST"])
def identify():
    """
    Nouveau : reçoit JSON { image: DataURL }
    Retourne { class_id, name, confidence }
    """
    data = request.get_json(force=True)
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "Pas d'image reçue"}), 400

    try:
        img = decode_image(img_b64)
        feats = extract_image_features(img)
        class_id = int(OBJECT_MODEL.predict(feats)[0])
        proba = float(np.max(OBJECT_MODEL.predict_proba(feats)))
        # Map d'exemple
        names = {0:"Bouteille",1:"Chaise",2:"Table",3:"Voiture",4:"Personne"}
        return jsonify({
            "class_id": class_id,
            "name": names.get(class_id, "Inconnu"),
            "confidence": round(proba, 3)
        }), 200

    except Exception as e:
        logger.error("❌ Erreur identify: %s", e)
        return jsonify({"error": "Impossible d'identifier l'image"}), 500

# Définir ces fonctions qui sont référencées mais manquantes dans le code original
def identify_urban_patterns(buildings, roads, water, landuse):
    # Simple implementation for the missing function
    patterns = ["Quartier résidentiel"]
    if len(water) > 0:
        patterns.append("Zone proche de l'eau")
    if len(roads) > 3:
        patterns.append("Quartier bien connecté")
    return patterns

def generate_recommendations(buildings, roads, water, landuse, patterns):
    # Simple implementation for the missing function
    recommendations = []
    if len(buildings) > 5:
        recommendations.append("Considérer l'ajout d'espaces verts")
    if len(roads) < 2:
        recommendations.append("Améliorer l'accès routier")
    return recommendations

# ——— Lancement ——————————————————————————————————————————

if __name__ == "__main__":
    logger.info("🚀 Démarrage du serveur IA Flask sur :5000")
    app.run(host="0.0.0.0", port=5000, debug=True)


