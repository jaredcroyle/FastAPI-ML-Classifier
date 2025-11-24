from fastapi import HTTPException

def validate_dna(seq: str):
    allowed = set("ACGTNacgtn")
    if any(c not in allowed for c in seq):
        raise HTTPException(
            status_code=400,
            detail="Sequence contains invalid DNA characters"
        )