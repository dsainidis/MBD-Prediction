import unicodedata

def remove_accents(input_str):



    """Removes langueage accents from strings
    
    Parameters
    --------
    input_str : string,
        The string with accents
                
    Returns
    ----------
    string: the string without accents 
    """
    
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])