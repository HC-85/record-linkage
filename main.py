# Load modules
#import warnings
#warnings.simplefilter("ignore", UserWarning)
import utils_dev
import time
from utils import *
from os import getlogin

def run_dev(nombre_catalogo_total, nombre_catalogo_comparar):
    usuario=getlogin()
    head_app = f"C:/Users/{usuario}/CEMEX/Gestión Inteligente & Sinergias Digitales - General/APPS/CATALOGO PIM"
    dict_unidades = crear_dict_unidades(f'{head_app}/DATOS/UM.csv')
    dict_equivalencias = crear_dict_equivalencias(f'{head_app}/DATOS/Diccionario de equivalencias.csv')
    ground_truth, repeated_codes = crear_ground_truth(f'{head_app}/DATOS/CÁTALOGO TOTAL\{nombre_catalogo_total}', dict_unidades, dict_equivalencias)
    cat_usuario = crear_catalogo_usuario(f'{head_app}/DATOS/CÁTALOGOS A COMPARAR\{nombre_catalogo_comparar}', ground_truth, dict_unidades, dict_equivalencias)
    matched_cat_usuario = matching(cat_usuario, ground_truth)
    results = clean_output(matched_cat_usuario, score_cutoff = 0)
    results.to_excel(f"{head_app}/RESULTADOS/Resultado {nombre_catalogo_comparar[:-4]}.xlsx", index=False)


if __name__ == "__main__":
    while True:
        archivos_comparar, archivos_catalogo_total = utils_dev.nombres_catalogos()
        utils_dev.sync_files_v2(archivos_comparar, archivos_catalogo_total)

        if archivos_catalogo_total.__len__() == 1:
            if archivos_comparar.__len__() != 0:
                hora = time.strftime("%H:%M:%S")
                print(
                    f"[{hora}] Se inicia un ciclo de trabajo sobre todos los archivos usando catalogo total {archivos_catalogo_total[0]}"
                )
                for archivo in archivos_comparar:
                    print(f"[{hora}] Se empieza a trabajar el archivo {archivo}")
                    try:
                       run_dev(nombre_catalogo_total=archivos_catalogo_total[0], nombre_catalogo_comparar=archivo)
                       hora = time.strftime("%H:%M:%S")
                       print(f"[{hora}] Se termino de trabajar el archivo {archivo}")
                    except:
                       hora = time.strftime("%H:%M:%S")
                       print(f"[{hora}] Hubo un error procesando el archivo {archivo}")
                hora = time.strftime("%H:%M:%S")
                print(
                    f"[{hora}] Se termina un ciclo de trabajo sobre todos los archivos."
                )
                time.sleep(60 * 60 * 4)
            else:
                hora = time.strftime("%H:%M:%S")
                print(f"[{hora}] No hay archivos en la carpeta para comparar")
                time.sleep(60)
        else:
            hora = time.strftime("%H:%M:%S")
            print(
                f"[{hora}] Debe haber solo un archivo en la carpeta de catalogo total. Se encontraron "
                + str(archivos_catalogo_total.__len__())
                + " archivos."
            )
            time.sleep(60)
