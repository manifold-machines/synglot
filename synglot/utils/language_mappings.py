"""
Language code mappings for different translation backends.

This module contains mappings between standard language codes and backend-specific formats.
"""

# NLLB language code mappings from standard ISO codes to NLLB format
NLLB_LANGUAGE_MAPPING = {
    'en': 'eng_Latn',
    'es': 'spa_Latn', 
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
    'tr': 'tur_Latn',
    'pl': 'pol_Latn',
    'nl': 'nld_Latn',
    'sv': 'swe_Latn',
    'da': 'dan_Latn',
    'no': 'nob_Latn',
    'fi': 'fin_Latn',
    'el': 'ell_Grek',
    'he': 'heb_Hebr',
    'th': 'tha_Thai',
    'vi': 'vie_Latn',
    'uk': 'ukr_Cyrl',
    'cs': 'ces_Latn',
    'hu': 'hun_Latn',
    'ro': 'ron_Latn',
    'bg': 'bul_Cyrl',
    'hr': 'hrv_Latn',
    'sk': 'slk_Latn',
    'sl': 'slv_Latn',
    'et': 'est_Latn',
    'lv': 'lav_Latn',
    'lt': 'lit_Latn',
    'mk': 'mkd_Cyrl',
    'id': 'ind_Latn',
    'ms': 'zsm_Latn',
    'bn': 'ben_Beng',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'ml': 'mal_Mlym',
    'kn': 'kan_Knda',
    'gu': 'guj_Gujr',
    'pa': 'pan_Guru',
    'ur': 'urd_Arab',
    'fa': 'pes_Arab',
    'sw': 'swh_Latn',
    'am': 'amh_Ethi',
    'ig': 'ibo_Latn',
    'yo': 'yor_Latn',
    'ha': 'hau_Latn',
    'zu': 'zul_Latn',
    'af': 'afr_Latn',
    'eu': 'eus_Latn',
    'ca': 'cat_Latn',
    'gl': 'glg_Latn',
    'cy': 'cym_Latn',
    'ga': 'gle_Latn',
    'is': 'isl_Latn',
    'mt': 'mlt_Latn',
    'sq': 'als_Latn',
    'be': 'bel_Cyrl',
    'az': 'azj_Latn',
    'ka': 'kat_Geor',
    'hy': 'hye_Armn',
    'kk': 'kaz_Cyrl',
    'ky': 'kir_Cyrl',
    'uz': 'uzn_Latn',
    'tg': 'tgk_Cyrl',
    'mn': 'khk_Cyrl',
    'ne': 'npi_Deva',
    'si': 'sin_Sinh',
    'my': 'mya_Mymr',
    'km': 'khm_Khmr',
    'lo': 'lao_Laoo'
}


def get_nllb_language_code(lang_code: str) -> str:
    """
    Map standard language code to NLLB format.
    
    Args:
        lang_code (str): Standard language code (e.g., 'en', 'fr')
        
    Returns:
        str: NLLB format language code (e.g., 'eng_Latn', 'fra_Latn')
        
    Note:
        If no mapping is found, returns a fallback format using Latin script.
    """
    mapped = NLLB_LANGUAGE_MAPPING.get(lang_code)
    if mapped:
        return mapped
    else:
        # If no mapping found, try to construct one with Latin script as default
        print(f"Warning: No NLLB mapping found for '{lang_code}', using '{lang_code}_Latn' as fallback")
        return f"{lang_code}_Latn"


def get_supported_nllb_languages():
    """
    Get list of supported language codes for NLLB.
    
    Returns:
        list: List of supported language codes
    """
    return list(NLLB_LANGUAGE_MAPPING.keys())


def is_nllb_language_supported(lang_code: str) -> bool:
    """
    Check if a language code is supported by NLLB.
    
    Args:
        lang_code (str): Language code to check
        
    Returns:
        bool: True if supported, False otherwise
    """
    return lang_code in NLLB_LANGUAGE_MAPPING 