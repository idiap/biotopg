# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

from typing import List

from pydantic import BaseModel


class BioEntity(BaseModel):
    id: str
    text: str


class BioHyperProposition(BaseModel):
    proposition: str
    entities: List[BioEntity]


class BioHyperPropositionList(BaseModel):
    biohyperpropositions: List[BioHyperProposition]


class HyperProposition(BaseModel):
    proposition: str
    entities: List[str]
