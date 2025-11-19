import os
import csv

def parse_arg_file(file_path: str) -> dict:
    """
    Parse un fichier .arg formaté comme :
    variable1 : X
    variable2 : Y, Z
    ...
    et retourne un dictionnaire avec les clés et valeurs.
    Si la valeur contient des virgules, elle est traitée comme une liste d'entiers.
    Sinon, essaie de convertir en entier, sinon garde comme string.
    """
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)  # split sur le premier :
                    key = key.strip()
                    value = value.strip()
                    if ',' in value:
                        # Traiter comme une liste d'entiers
                        result[key] = [int(item.strip()) for item in value.split(',')]
                    else:
                        try:
                            result[key] = int(value)
                        except ValueError:
                            result[key] = value
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return {}
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return {}
    return result

def load_best_by_error_rate(csv_path: str) -> dict:
    """Charge le mapping Error-Rate -> (Threshold, Distance) depuis un CSV.
    Retourne un dict: {float(error_rate): {'threshold': float, 'distance': float}}
    """
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                er = float(row.get('Error-Rate'))
                thr = float(row.get('Threshold'))
                dist = float(row.get('Distance'))
            except Exception:
                continue
            mapping[er] = {'threshold': thr, 'distance': dist}
    return mapping
