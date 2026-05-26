import os
import shutil

images_dir = "/Users/ghabrielcalado/Documents/deeplearning_pets/images"

for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Extrai o nome da raça (tudo antes do último _número)
    parts = filename.rsplit("_", 1)
    if len(parts) != 2:
        print(f"Ignorado (nome inesperado): {filename}")
        continue

    breed = parts[0]  # ex: "Abyssinian"

    # Cria a subpasta se não existir
    breed_dir = os.path.join(images_dir, breed)
    os.makedirs(breed_dir, exist_ok=True)

    # Move a imagem
    src = os.path.join(images_dir, filename)
    dst = os.path.join(breed_dir, filename)
    shutil.move(src, dst)

print("Dataset organizado com sucesso!")