# HarveyNER
We introduce a new dataset HarveyNER with fne-grained locations annotated in tweets. This dataset presents unique challenges and characterizes many complex and long location mentions in informal descriptions. We built strong baseline models using Curriculum Learning and experimented with different heuristic curricula to better recognize diffcult location mentions. 
HarveyNER focuses on the coordinate-oriented locations so we mainly annotate Point that can be precisely pinned to a map and Area that occupies a small polygon of a map. Considering that some disasters can affect line-like objects (e.g., a food can affect the neighbors of a whole river), we also include Road and River types.
* Points: denote an exact location that a geocoordinate can be assigned. E.g., a uniquely named building, intersections of roads or rivers.
* Areas: denote geographical entities such as city subdivisions, neighborhoods, etc.
* Roads: denote a road or a section of a road.
* Rivers: denote a river or a section of a river.

# Statistics of HarveyNER
|    Data Split    | Train | Valid |  Test | Total |
|:----------------:|:-----:|:-----:|:-----:|:-----:|
|    All Tweets    | 3,967 | 1,301 | 1,303 | 6,571 |
|  Tweet w/ Entity | 1,087 |  366  |  353  | 1,806 |
| Tweet w/o Entity | 2,880 |  935  |  950  | 4,765 |
|  All Entity Type | 1,581 |  523  |  500  | 2,604 |
|       Point      |  591  |  206  |  202  |  999  |
|       Area       |  715  |  236  |  212  | 1,163 |
|       Road       |  158  |   51  |   57  |  266  |
|       River      |  117  |   30  |   29  |  176  |
