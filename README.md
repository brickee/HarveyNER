# HarveyNER
We introduce a new dataset HarveyNER with fne-grained locations annotated in tweets. This dataset presents unique challenges and characterizes many complex and long location mentions in informal descriptions. We built strong baseline models using Curriculum Learning and experimented with different heuristic curricula to better recognize diffcult location mentions. 
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
