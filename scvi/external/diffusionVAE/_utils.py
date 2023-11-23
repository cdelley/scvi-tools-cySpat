import numpy as np
from natsort import natsorted
from typing import Dict

class encoding_converters():
    """
    A utility class for converting spatial encoding print files into structured
    formats for data processing.

    Designed for spatial transcriptomics files, this class translates encoding
    files into formats suitable for analysis.

    Methods:
        from_scienion_print_file: Converts Scienion print files to structured
                                  dictionary format.
    """
    
    @staticmethod
    def from_scienion_print_file(
        path: str
    ) -> Dict:
        """
        Reads a Scienion print file (CSV/TSV) and extracts spatial encoding.

        Processes files to extract and reformat spatial encoding data, handling
        Scienion print files' specific formats.

        Args:
            path (str): File path to the Scienion print file, CSV or TSV.

        Returns:
            Dict: A dictionary with spatial encoding data, including:
                - "numerical_encoding" (List[Tuple[int, int]]): Numeric encodings.
                - "encoding_x_position" (np.ndarray): X positions array.
                - "encoding_y_position" (np.ndarray): Y positions array.
                - "encodings_str" (List[Tuple[str, str]]): Original string encodings.
                - "str_to_num_convert" (Dict[str, int]): String to numeric mapping.
        """
        with open(path, 'r') as fin:
            if path.split('.')[-1] == 'csv':
                spots = [l.strip().split('"')[1::2] for l in fin]
            elif path.split('.')[-1] == 'tsv':
                spots = [l.strip().split('\t') for l in fin]
        
        # spots are strings of form: '2/50 1A2,1H7'
        encodings_str = [] 
        encoding_x = [] 
        encoding_y = []
        labels = []
        for row in spots:
            for s in row:
                loc, label = s.split()
                x = int(loc.split('/')[1])
                y = int(loc.split('/')[0])
                encoding_x.append(x)
                encoding_y.append(y)
                l1, l2 = label.split(',')
                encodings_str.append((l1, l2))
                labels.append(l1)
                labels.append(l2)
        
        # use of natsort to get ordering: 1A1, 1A2, ..., 1A10, 1A11
        labels = natsorted(list(set(labels)))
        label_convert = {l: n for (n,l) in enumerate(labels)}
        encodings = []
        for l1, l2 in encodings_str:
            encodings.append((label_convert[l1],label_convert[l2]))
        encoding_x = np.array(encoding_x) - np.mean(encoding_x)
        encoding_y = np.array(encoding_y) - np.mean(encoding_y)
        
        out_dict = {
            "numerical_encoding": encodings,
            "encoding_x_position": encoding_x,
            "encoding_y_position": encoding_y,
            "encodings_str": encodings_str,
            "str_to_num_convert": label_convert,
        }
        return out_dict
