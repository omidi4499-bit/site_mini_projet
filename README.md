# Mesure d'objet par photo

Application web qui estime les dimensions d'un objet a partir d'une seule image, sans reference manuelle.

## Utilisation

1. Ouvre l'application.
2. Choisis une image (ou ouvre la camera).
3. Place 2 points pour la largeur et 2 points pour la hauteur.
4. Selectionne l'unite.
5. Lance la mesure.

## Lancement

```bash
python app.py
```

Puis ouvre:

- http://127.0.0.1:5000

## Configuration du service d'analyse

Le backend consomme un service d'analyse externe. Configure:

```powershell
$env:HF_TOKEN="ton_token_huggingface"
# optionnel
$env:HF_DEPTH_API_URL="https://api-inference.huggingface.co/models/Intel/dpt-hybrid-midas"
python app.py
```

## Notes

- Les resultats sont des estimations.
- Une photo nette et frontale ameliore la precision.
