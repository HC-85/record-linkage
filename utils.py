from rapidfuzz import fuzz, process
import pandas as pd
import numpy as np
import re
import unidecode
from tqdm import tqdm
from collections import Counter


def remover_sa_de_cv(df, col):
    for n, item in enumerate(df[col]):
        try:
            df.loc[n, col] = re.search('(.+) S(?:\.|)A(?:\.|) DE C(?:\.|)V(?:\.|)', item).groups()[0]
        except:
            pass
    return df


def smart_replace(objetivo, remplazo, string, max_reps = 3):
    flag = True
    reps = 0
    while flag and (reps<=max_reps):
        reps += 1
        re_try = re.search(f'((^|[^A-Z]){objetivo}([^A-Z]|$))', string)
        if re_try is not None:
            string = string.replace(re_try[0], re_try[0].replace(objetivo, remplazo))
        else:
            flag = False
    return string


def crear_dict_unidades(archivo_entrada):
    codigos_unidades = pd.read_csv(archivo_entrada, usecols = ['Código', 'Nombre'])
    codigos_unidades['Código'] = codigos_unidades['Código'].apply(lambda x: str.upper(x))
    codigos_unidades['Nombre'] = codigos_unidades['Nombre'].apply(lambda x: str.upper(x))

    dict_unidades = [*(codigos_unidades.set_index('Código').to_dict().values())][0]

    # Correcciones manuales
    dict_unidades['KILOS'] = 'KG'
    dict_unidades['KILOGRAMOS'] = 'KG'
    dict_unidades['BG'] = 'BAG'
    dict_unidades['BOLSA'] = 'BAG'
    dict_unidades['BOX'] = 'CAJA'
    dict_unidades['CUBETA 19L'] = 'BCK'
    dict_unidades['GALON 4LT'] = 'GLL'
    dict_unidades['LITRO 1LT'] = 'L'
    dict_unidades['LT'] = 'L'
    dict_unidades['TN'] = 'TO'
    dict_unidades['CENTIMETROS'] = 'CM'
    dict_unidades['CENTÍMETROS'] = 'CM'
    dict_unidades['JEGO'] = 'JGO'
    dict_unidades['EAE'] = 'EA'
    dict_unidades['EZ'] = 'EA'
    dict_unidades['A'] = 'EA'
    dict_unidades['GALÓN'] = 'GLL'
    dict_unidades['PZ'] = 'EA'
    dict_unidades['MT'] = 'M'
    dict_unidades['ML'] = 'M'

    return dict_unidades


def crear_dict_equivalencias(archivo_entrada):
    equivalencias = pd.read_csv(archivo_entrada, usecols = ['PALABRA A REMPLAZAR', 'REMPLAZO'])
    equivalencias['PALABRA A REMPLAZAR'] = equivalencias['PALABRA A REMPLAZAR'].apply(lambda x: str.upper(x))
    equivalencias['REMPLAZO'] = equivalencias['REMPLAZO'].apply(lambda x: str.upper(x))
    dict_equivalencias = [*(equivalencias.set_index('PALABRA A REMPLAZAR').to_dict().values())][0]

    return dict_equivalencias


def crear_ground_truth(archivo_entrada, dict_unidades, dict_equivalencias):
    cols = ['CODIGO_SKU', 'NOMBRE_SKU', 'CODIGO_FABRICANTE', 'FABRICANTE', 'MARCA', 'Unidad de Venta']
    GT = pd.read_csv(archivo_entrada, 
    usecols=cols, dtype = {'CODIGO_SKU':str, 'NOMBRE_SKU': str, 'CODIGO_FABRICANTE':str, 'FABRICANTE': str, 'MARCA': str, 'Unidad de Venta': str}, 
    skipfooter=0, engine = 'python')

    GT = GT.rename(columns={'Unidad de Venta': 'UNIDAD'})

    # Remover espacios en blanco, pasar a mayúsculas. Mantener una copia intacta
    for col in cols: 
        GT[f'{col} (ORIGINAL)'] = GT[col]
        GT.loc[:, col] = GT[col].apply(lambda x: unidecode.unidecode(str.upper(str(x)).strip()))

    # Estandarizar columna de unidades
    GT.loc[:, 'UNIDAD'] = GT['UNIDAD'].apply(lambda x: dict_unidades[x] if x in dict_unidades.keys() else x)

    # Estandarizar las unidades
    for n, desc in enumerate(GT['NOMBRE_SKU']):
        for cod_unidad, unidad in dict_unidades.items():
            re_try = re.search(f'([^A-Z]{cod_unidad}([^A-Z]|$))', desc)
            if re_try is not None:
                sub_re_try = re_try.groups()[0].strip().replace(cod_unidad, f' {unidad}')
                desc = desc.replace(re_try.groups()[0], sub_re_try)
                GT.loc[n, 'NOMBRE_SKU'] = desc

    # Estandarizar columna de codigo de fabricante
    dict_cod_fab = {'NAN': 'SIN CODIGO'}
    GT.loc[:, 'CODIGO_FABRICANTE'] = GT['CODIGO_FABRICANTE'].apply(lambda x: dict_cod_fab[x] if x in dict_cod_fab.keys() else x)

    # Estandarizar columna de marca / fabricante
    dict_gen = {'GENERICOS': 'GENERICO'}
    GT.loc[:, 'FABRICANTE'] = GT['FABRICANTE'].apply(lambda x: dict_gen[x] if x in dict_gen.keys() else x)
    GT.loc[:, 'MARCA'] = GT['MARCA'].apply(lambda x: dict_gen[x] if x in dict_gen.keys() else x)

    # Normalizar los fabricantes y marcas para no tener SA DE CV
    GT = remover_sa_de_cv(GT, 'FABRICANTE')
    GT = remover_sa_de_cv(GT, 'MARCA')

    # Todo generico debe tener SIN CODIGO
    GT['CODIGO_FABRICANTE'] = GT[['CODIGO_FABRICANTE', 'FABRICANTE', 'MARCA']].apply(lambda x: 'SIN CODIGO' if ((x[1]=='GENERICO')|(x[2]=='GENERICO')) else x[0], axis = 1)

    # Eliminamos las entradas repetidas 
    GT = GT.drop_duplicates().reset_index(drop=True)

    # Crear set de marcas y fabricantes
    set_marca_fabri = set(GT['MARCA']).union(set(GT['FABRICANTE']))

    # Crear set de unidades
    set_unidades = set(GT['UNIDAD'])

    # Checar si tiene marca/fabricante en la descripción 
    for n, desc in enumerate(GT['NOMBRE_SKU']):
        desc_limpia = desc
        try:
            marca_tent = re.match('([^,]+), ', desc).groups()[0]
            marca_desc = marca_tent if marca_tent in set_marca_fabri else ''
        except:
            try:
                marca_col = GT.loc[n, 'MARCA']
                marca_desc = re.search(f'({marca_col})', desc).groups()[0]
            except:
                try:
                    marca_col = GT.loc[n, 'FABRICANTE']
                    marca_desc = re.search(f'({marca_col})', desc).groups()[0]
                except:
                    marca_desc = ''
        GT.loc[n, 'MARCA_DESC'] = marca_desc
        
        if marca_desc != '':
            desc_limpia = re.sub(marca_desc, '', desc_limpia)
            
        # Checar si tiene unidades en la descripción
        try: 
            unidad_tent = re.search(', ([^,]+)$', desc).groups()[0]
            if re.match('VENTA POR (.+)', unidad_tent) is not None:
                unidad_desc = re.match('VENTA POR (.+)', unidad_tent).groups()[0]
                desc_limpia = re.sub(f'VENTA POR {unidad_desc}', '', desc_limpia)
            else:
                unidad_desc = unidad_tent if unidad_tent in set_unidades else ''
        except:
            unidad_desc = ''
        GT.loc[n, 'UNIDAD_DESC'] = unidad_desc
        
        if unidad_desc != '':
            desc_limpia = re.sub(unidad_desc, '', desc_limpia)

        desc_limpia = re.sub(',', '', desc_limpia).strip()    

        # Unidades solas -> (1)
        for unidad in set(dict_unidades.values()):
            re_try = re.search(f'([^0-9] {unidad}( |$))', desc_limpia)
            if re_try is not None:
                sub_re_try = re_try.groups()[0].strip().replace(unidad, f'1 {unidad}')
                desc_limpia = desc_limpia.replace(re_try.groups()[0], sub_re_try)
        
        # Las fracciones no deben de superar 1
        fracs_ok = [False]
        try:
            while any([not f for f in fracs_ok]):
                fracs = re.findall('([0-9]+/[0-9]+)', desc_limpia)
                if len(fracs) == 0:
                    fracs_ok = [True]
                else:
                    fracs_ok = []
                    for frac in fracs:
                        num = re.search('([0-9]+)/', frac).groups()[0]
                        den = re.search('/([0-9]+)', frac).groups()[0]
                        if float(num)>float(den):
                            fracs_ok.append(False)                
                            desc_limpia = desc_limpia.replace(frac, f'{frac[0]} {frac[1:]}')
                        else:
                            fracs_ok.append(True)
        except:
            fracs_ok         
        
        GT.loc[n, 'DESC_LIMPIA'] = desc_limpia

    # Remplazar equivalencias
    for objetivo, remplazo in dict_equivalencias.items():
            GT['DESC_LIMPIA'] = GT['DESC_LIMPIA'].apply(lambda x: smart_replace(objetivo, remplazo, x))

    # Si existen mas de dos articulos con el mismo NOMBRE_SKU, se conserva el primero pero se guarda el resto
    tmp = GT.groupby('NOMBRE_SKU')['CODIGO_SKU'].apply(list).reset_index(name = 'CODES')
    repeated_codes = tmp[tmp.CODES.apply(len)>1]

    GT = GT.groupby('NOMBRE_SKU').first().reset_index()

    return GT, repeated_codes


def crear_catalogo_usuario(path_catalogo_usuario, ground_truth, dict_unidades, dict_equivalencias, test = False):
    if test: 
        col_types = {'CÓDIGO PIM': str, 'DESCRIPCIÓN PIM': str, 'CÓDIGO (ÚNICO)': str, 'DESCRIPCIÓN': str, 'UNIDAD DE MEDIDA': str, 
        'NÚMERO DE FABRICANTE': str, 'MARCA':str , 'FABRICANTE': str}
        col_renames = {'CÓDIGO PIM': 'CODIGO_PIM_TARGET','CÓDIGO (ÚNICO)': 'CODIGO_UNICO', 'NÚMERO DE FABRICANTE': 'CODIGO_FABRICANTE', 
        'DESCRIPCIÓN': 'DESC_USUARIO', 'UNIDAD DE MEDIDA': 'UNIDAD', 'DESCRIPCIÓN PIM': 'TARGET'}
    else:
        col_types = {'CÓDIGO (ÚNICO)': str, 'DESCRIPCIÓN': str, 'UNIDAD DE MEDIDA': str, 'NÚMERO DE FABRICANTE': str, 'MARCA':str, 'FABRICANTE': str}
        col_renames = {'CÓDIGO (ÚNICO)': 'CODIGO_UNICO', 'NÚMERO DE FABRICANTE': 'CODIGO_FABRICANTE', 'DESCRIPCIÓN': 'DESC_USUARIO', 'UNIDAD DE MEDIDA': 'UNIDAD'}
        
    try:
        cat_usuario = pd.read_csv(path_catalogo_usuario, usecols = list(col_types.keys()), dtype = col_types)
    except:
        cat_usuario = pd.read_csv(path_catalogo_usuario, usecols = list(col_types.keys()), dtype = col_types, encoding='latin-1')

    cat_usuario = cat_usuario.rename(columns=col_renames)
    
    # Definir sets generales
    set_items = set(ground_truth['DESC_LIMPIA'])
    set_marca_fabri = set(ground_truth['MARCA']).union(set(ground_truth['FABRICANTE']))
    set_unidades = set(ground_truth['UNIDAD'])
    
    # Remover acentos y espacios en blanco, pasar a mayúsculas.
    for col in cat_usuario.columns:
        cat_usuario[f'{col} (ORIGINAL)'] = cat_usuario[col]
        cat_usuario.loc[:, col] = cat_usuario[col].apply(lambda x: unidecode.unidecode(str.upper(str(x)).strip()))

    # Estandarizar columnas marca / fabricante
    dict_gen = {'GENERICOS': 'GENERICO'}
    cat_usuario.loc[:, 'FABRICANTE'] = cat_usuario['FABRICANTE'].apply(lambda x: dict_gen[x] if x in dict_gen.keys() else x)
    cat_usuario.loc[:, 'MARCA'] = cat_usuario['MARCA'].apply(lambda x: dict_gen[x] if x in dict_gen.keys() else x)

    # Normalizar los fabricantes y marcas para no tener SA DE CV
    cat_usuario = remover_sa_de_cv(cat_usuario, 'FABRICANTE')
    cat_usuario = remover_sa_de_cv(cat_usuario, 'MARCA')

    # Estandarizar columna codigo de fabricante
    dict_sincodigo = {'SIN CODGIO': 'SIN CODIGO', 'NAN': 'SIN CODIGO', '': 'SIN CODIGO'}
    cat_usuario.loc[:, 'CODIGO_FABRICANTE'] = cat_usuario['CODIGO_FABRICANTE'].apply(lambda x: dict_sincodigo[x] if x in dict_sincodigo.keys() else x)

    # Todo generico debe tener SIN CODIGO
    cat_usuario['CODIGO_FABRICANTE'] = cat_usuario[['CODIGO_FABRICANTE', 'FABRICANTE', 'MARCA']].apply(lambda x: 'SIN CODIGO' if ((x[1]=='GENERICO')|(x[2]=='GENERICO')) else x[0], axis = 1)

    # Estandarizar columna de unidades
    cat_usuario.loc[:, 'UNIDAD'] = cat_usuario['UNIDAD'].apply(lambda x: dict_unidades[x] if x in dict_unidades.keys() else x)
    
    # Remplazar equivalencias
    for objetivo, remplazo in dict_equivalencias.items():
            cat_usuario['DESC_USUARIO'] = cat_usuario['DESC_USUARIO'].apply(lambda x: smart_replace(objetivo, remplazo, x))
            
    for n, desc_limpia in enumerate(cat_usuario['DESC_USUARIO']):
        # Las fracciones no deben de superar 1
        fracs_ok = [False]
        while any([not f for f in fracs_ok]):
            fracs = re.findall('([0-9]+/[0-9]+)', desc_limpia)
            if len(fracs) == 0:
                fracs_ok = [True]
            else:
                fracs_ok = []
                for frac in fracs:
                    num = re.search('([0-9]+)/', frac).groups()[0]
                    den = re.search('/([0-9]+)', frac).groups()[0]
                    if float(num)>float(den):
                        fracs_ok.append(False)                
                        desc_limpia = desc_limpia.replace(frac, f'{frac[0]} {frac[1:]}')
                    else:
                        fracs_ok.append(True)
        cat_usuario.loc[n, 'DESC_USUARIO'] = desc_limpia

    # Eliminamos las entradas repetidas 
    cat_usuario = cat_usuario.drop_duplicates().reset_index(drop=True)

    if test:
        # !!!!!!!!!! Algunos targets solo dicen "NAN", se eliminan por ahora !!!!!!!!!!!!!!!
        cat_usuario = cat_usuario.query('TARGET != "NAN"').reset_index(drop=True)

        # Los targets tienen marca y unidades, utilizar las descripciones limpias del catalogo de referencia en su lugar
        cat_usuario = \
            cat_usuario.set_index('TARGET')\
            .join(ground_truth.set_index('NOMBRE_SKU')['DESC_LIMPIA'])\
            .reset_index(names='TARGET')\
            .rename(columns = {'DESC_LIMPIA': 'TARGET_LIMPIO'})

        # Algunos targets no tienen match con el catalogo de referencia
        sub = cat_usuario.query('TARGET_LIMPIO.isnull()')
        for ind in sub.index:
            target = cat_usuario.loc[ind, 'TARGET']

            # Algunos tienen el codigo SKU por error, los remplazamos por su descripcion correspondiente
            try:
                codigo_sku = re.search('^\d+$', target)[0]
                if len(codigo_sku)<10:
                    codigo_sku = (10-len(codigo_sku))*'0' + codigo_sku
                cat_usuario.loc[ind, 'TARGET_LIMPIO'] = ground_truth.query('CODIGO_SKU == @codigo_sku')['DESC_LIMPIA'].values[0]
            
            except: # Limpiar la descripcion para comparar con fuzzy
                target_limpio = target
                for cod_unidad, unidad in dict_unidades.items():
                    re_try = re.search(f'([^A-Z]{cod_unidad}([^A-Z]|$))', target_limpio)
                    if re_try is not None:
                        sub_re_try = re_try.groups()[0].strip().replace(cod_unidad, f' {unidad}')
                        target_limpio = target_limpio.replace(re_try.groups()[0], sub_re_try)

                # Checar si tiene marca/fabricante
                try:
                    marca_tent = re.match('([^,]+), ', target).groups()[0]
                    marca_target = marca_tent if marca_tent in set_marca_fabri else ''
                except:
                    marca_target = ''
                
                if marca_target != '':
                    target_limpio = re.sub(marca_target, '', target_limpio)
                    
                # Checar si tiene unidades
                try: 
                    unidad_tent = re.search(', ([^,]+)$', target_limpio).groups()[0]
                    if re.match('VENTA POR (.+)', unidad_tent) is not None:
                        unidad_target = re.match('VENTA POR (.+)', unidad_tent).groups()[0]
                        target_limpio = re.sub(f'VENTA POR {unidad_target}', '', target_limpio)
                    else:
                        unidad_target = unidad_tent if unidad_tent in set_unidades else ''
                except:
                    unidad_target = ''
                
                if unidad_target != '':
                    if re.search(f'{unidad_target}$', target_limpio) is not None:
                        target_limpio = re.sub(f'{unidad_target}$', '', target_limpio)
                    #else:
                    #    target_limpia = re.sub(unidad_target, '', target_limpio)

                target_limpio = re.sub(',', '', target_limpio).strip()    
                
                Q_guess = process.extractOne(target_limpio, scorer=fuzz.QRatio, choices=set_items)
                cat_usuario.loc[ind, 'TARGET_LIMPIO'] = Q_guess[0]

    # Se asume por ahora que la descripcion del usuario no tiene ni marca ni unidades
    return cat_usuario


def matching(cat_usuario, ground_truth, n_matches = 3, test = False, weights = {'similarity': 35, 'marca_fabri': 20, 'unidad': 5, 'nums': 20, 'cod_fab': 50}, similarity_scorer = fuzz.QRatio, similarity_returns = 100, similarity_cutoff = 40):
    # Agregar banderas como retro para el desarrollador
    cat_usuario['FLAGS'] = np.empty((len(cat_usuario), 0)).tolist()
    set_choices = set(ground_truth['DESC_LIMPIA'])
    og_weights = weights.copy()

    if type(test) == list:
        cat_usuario = cat_usuario.query('`DESC_USUARIO (ORIGINAL)` in @test').reset_index()

    for n, item in enumerate(tqdm(cat_usuario['DESC_USUARIO'])):
        total_weight = sum(sorted(weights.values()))
        for key, value in weights.items():
            weights[key] = value/total_weight
        if (type(test) == bool):
            if test:
                target = cat_usuario.loc[n, 'TARGET_LIMPIO']
        
        cod_fab = cat_usuario.loc[n, 'CODIGO_FABRICANTE']
        marca_fabri = cat_usuario.loc[n, ['MARCA', 'FABRICANTE']].values
        unidad = cat_usuario.loc[n, 'UNIDAD']

        set_choices_cf = set(ground_truth.query('CODIGO_FABRICANTE == @cod_fab')['DESC_LIMPIA'])
        set_choices_mf = set(ground_truth.query('(MARCA in @marca_fabri) | (FABRICANTE in @marca_fabri)')['DESC_LIMPIA'])
        set_choices_uni = set(ground_truth.query('UNIDAD in @unidad')['DESC_LIMPIA'])
        los_setos = [set_choices, set_choices_mf, set_choices_uni, set_choices_cf]

        if (type(test) == bool):
            if test:
                if target not in set_choices:
                    cat_usuario.loc[n, 'FLAGS'].append('TARGET NOT IN ITEMS SET')
                if target not in set_choices_cf:
                    cat_usuario.loc[n, 'FLAGS'].append('TARGET NOT IN CF ITEMS SET')
                if target not in set_choices_mf:
                    cat_usuario.loc[n, 'FLAGS'].append('TARGET NOT IN MF ITEMS SET')
                if target not in set_choices_uni:
                    cat_usuario.loc[n, 'FLAGS'].append('TARGET NOT IN UNI ITEMS SET')

        # Se intenta hacer match con el codigo de fabricante
        if cod_fab == 'SIN CODIGO':
            cat_usuario.loc[n, 'FLAGS'].append('NO COD FAB')
            weights = og_weights.copy()
            weights['similarity'] = 100
            weights['marca_fabri'] = 80
            weights['unidad'] = 50
            weights['nums'] = 80
            weights['cod_fab'] = 25
            tmp_weights = weights.copy()
            total_weight = sum(sorted(weights.values()))
            for key, value in weights.items():
                weights[key] = value/total_weight
            
            #los_setos.remove(set_choices_cf)
        else:
            tmp_weights = weights.copy()
        
        preds = []
        preds_sets = []
        for choices in los_setos:
            _preds = process.extract(item, choices=choices, scorer=similarity_scorer, limit = similarity_returns)
            preds.append(dict([[pred[0], pred[1]*weights['similarity']] for pred in _preds]))
            preds_sets.append(set([pred[0] for pred in _preds]))

        for pred in (preds_sets[0] & preds_sets[1]):
            preds[0][pred] = preds[0][pred] + 100*weights['marca_fabri']
        for pred in (preds_sets[1] - preds_sets[0]):
            preds[0][pred] = preds[1][pred]

        for pred in (preds_sets[0] & preds_sets[2]):
            preds[0][pred] = preds[0][pred] + 100*weights['unidad']
        for pred in (preds_sets[2] - preds_sets[0]):
            preds[0][pred] = preds[2][pred]

        if weights['cod_fab'] != 0:
            for pred in (preds_sets[0] & preds_sets[3]):
                preds[0][pred] = preds[0][pred] + 100*weights['cod_fab']
            for pred in (preds_sets[3] - preds_sets[0]):
                preds[0][pred] = preds[3][pred] 
                
        preds = preds[0]

        preds = pd.DataFrame.from_dict(data = {'PRED': preds.keys(), 'SCORE': preds.values()})

        nums_item = re.findall('[0-9]+\.*/*[0-9]*', item)
        if len(nums_item)==0:
            preds['SCORE'] = preds['SCORE']*total_weight
            weights = tmp_weights.copy()
            weights['nums'] = 0
            total_weight = sum(sorted(weights.values()))
            for key, value in weights.items():
                weights[key] = value/total_weight
            preds['SCORE'] = preds['SCORE']/total_weight
        else:
            preds['TEMP'] = preds['PRED'].apply(lambda x: Counter(re.findall('[0-9]+\.*/*[0-9]*', x)))
            preds['SCORE'] = preds[['PRED', 'SCORE', 'TEMP']].apply(lambda x: x[1] + 100*weights['nums']*(len(sorted( (x[2] & Counter(nums_item)).elements() ))\
                - abs( len(sorted(x[2].elements())) - len(sorted(Counter(nums_item).elements())) ))/len(sorted(Counter(nums_item).elements())), axis = 1)

        preds = preds.sort_values('SCORE', ascending=False).iloc[:n_matches, :].reset_index(drop=True)

        for m in range(n_matches):
            articulo_limpio = preds.loc[m, 'PRED']
            articulo = ground_truth.query('DESC_LIMPIA == @articulo_limpio')
            cat_usuario.loc[n, f'ARTICULO_SUGERIDO_ORIGINAL ({m+1})'] = articulo['NOMBRE_SKU (ORIGINAL)'].values[0]
            cat_usuario.loc[n, f'ARTICULO_SUGERIDO ({m+1})'] = articulo['DESC_LIMPIA'].values[0]
            cat_usuario.loc[n, f'ARTICULO_LIMP ({m+1})'] = articulo_limpio
            cat_usuario.loc[n, f'COD_PIM_SUGERIDO ({m+1})'] = articulo['CODIGO_SKU (ORIGINAL)'].values[0]
            cat_usuario.loc[n, f'PUNTAJE ({m+1})'] = preds.loc[m, 'SCORE']
            cat_usuario.loc[n, f'MARCA_FABRI_MATCH ({m+1})'] = articulo_limpio in set_choices_mf
            cat_usuario.loc[n, f'UNIDAD_MATCH ({m+1})'] = articulo_limpio in set_choices_uni
            cat_usuario.loc[n, f'COD_FAB_MATCH ({m+1})'] = articulo_limpio in set_choices_cf
            if (type(test) == bool):
                if test:
                    cat_usuario.loc[n, f'is_match ({m+1})'] = cat_usuario['TARGET_LIMPIO'] == articulo_limpio

    return cat_usuario


def clean_output(cat_usuario, score_cutoff, test = False):
    if test:
        cols = ['DESC_USUARIO', 'DESC_USUARIO', 'TARGET_LIMPIO', 'FLAGS']
    else:
        cols = ['CODIGO_UNICO (ORIGINAL)', 'DESC_USUARIO (ORIGINAL)', 'DESC_USUARIO', 'CODIGO_FABRICANTE (ORIGINAL)', 'FABRICANTE (ORIGINAL)', 'MARCA (ORIGINAL)', 'UNIDAD (ORIGINAL)']


    n_matches = int(sum([(1 if re.search('.\([0-9]+\)' , column) is not None else 0) for column in cat_usuario.columns])/8)

    for n in range(n_matches):    
        cols.append(f'COD_PIM_SUGERIDO ({n+1})')
        cols.append(f'ARTICULO_SUGERIDO_ORIGINAL ({n+1})')
        cols.append(f'ARTICULO_SUGERIDO ({n+1})')
        cols.append(f'ARTICULO_LIMP ({n+1})')
        cols.append(f'PUNTAJE ({n+1})')
        cols.append(f'MARCA_FABRI_MATCH ({n+1})')
        cols.append(f'UNIDAD_MATCH ({n+1})')
        cols.append(f'COD_FAB_MATCH ({n+1})')

    cat_usuario = cat_usuario[cols]

    for col in cols:
        try:
            og = re.match('(.+) \(ORIGINAL\)$', col).groups()[0]
            cat_usuario = cat_usuario.rename(columns = {col: og})
        except:
            pass

    for n in range(n_matches):    
        cat_usuario[f'ARTICULO_SUGERIDO ({n+1})'] = cat_usuario[[f'ARTICULO_SUGERIDO ({n+1})', f'PUNTAJE ({n+1})']].apply(lambda x: x[0] if (x[1] > score_cutoff) else '-- PUNTAJE MUY BAJO --', axis = 1)

    return cat_usuario