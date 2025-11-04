from PIL import Image
import os, sys

root = os.path.dirname(os.path.abspath(__file__))
png = os.path.join(root, 'logo.png')
ico = os.path.join(root, 'logo.ico')
if not os.path.exists(png):
    print('No se encontró logo.png en', root)
    sys.exit(1)
img = Image.open(png).convert('RGBA')
sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]
img.save(ico, sizes=sizes)
print('Icono generado:', ico)
