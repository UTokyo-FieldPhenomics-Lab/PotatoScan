import json
import os
import numpy as np

class MyEncoder(json.JSONEncoder):
    # The original json package doesn't compatible to numpy object, add this compatible encoder to it.
    # usage: json.dump(..., cls=MyEncoder)
    # references: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
def dict2json(data_dict, json_path, indent=None, encoding='utf-8'):
    """Save dict object to the same structure json file
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json`` 
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> a = {"test": {"rua": np.asarray([[12, 34], [45, 56]])}, "hua":[np.int32(34), np.float64(34.567)]}
        >>> idp.jsonfile.dict2json(a, "/path/to/save/json_file.json")

    .. note:: 

        Dict without indient:

        .. code-block:: python

            >>> print(json.dumps(data), indent=0)
            {"age": 4, "name": "niuniuche", "attribute": "toy"}

        Dict with 4 space as indient:

        .. code-block:: python

            >>> print(json.dumps(data,indent=4))
            {
                "age": 4,
                "name": "niuniuche",
                "attribute": "toy"
            }

    See also
    --------
    easyidp.jsonfile.write_json, easyidp.jsonfile.save_json
    
    """
    json_path = str(json_path)
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w', encoding=encoding) as result_file:
            json.dump(data_dict, result_file, ensure_ascii=False, cls=MyEncoder, indent=indent)