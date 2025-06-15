# common/hoquery.py
# Query keys and collate function for HO3DTrialDataset outputs

from main.utils.misc import ImmutableClass
from typing import List, Dict, Any

class TrialQueries(metaclass=ImmutableClass):
    """
    Keys for the dictionary returned by HO3DTrialDataset and load_trial_data.
    Must match the keys produced by load_trial_data().
    """
    SCENE               = 'scene'
    TRIAL               = 'trial'

    TRANSLATION         = 'trans'      # simulation translations
    ROTATION            = 'rot'        # simulation rotations
    CONTACT             = 'contact'    # contact info per frame

    LHAND_VERTS         = 'lhand_verts'
    LHAND_FACES         = 'lhand_faces'
    RHAND_VERTS         = 'rhand_verts'
    RHAND_FACES         = 'rhand_faces'
    OBJ_VERTS           = 'obj_verts'
    OBJ_FACES           = 'obj_faces'

    VHACD_LHAND_PARTS   = 'vhacd_lhand_parts'
    VHACD_RHAND_PARTS   = 'vhacd_rhand_parts'
    VHACD_OBJ_PARTS     = 'vhacd_obj_parts'


def trial_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collate function for trial-level dataset.
    - Filters out any None entries (skipped trials).
    - Returns the list of trial dicts as a batch.
    """
    # drop any None items returned when a trial failed to load
    batch = [b for b in batch if b is not None]
    return batch
