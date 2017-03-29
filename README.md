Ecclesia
=======================
The goal of this project is to quantify how representative current and proposed districts are  by comparing their simulated outcomes to what we would expect if we were to have a representative set of districts. The first step of this is to  Two principal types of districts will be presented:
1. Districts as established by the state of Wisconsin
2. Algorithmically generated districts.
  * Currently, districts have been created by a version of [K means](https://elki-project.github.io/tutorial/same-size_k_means) using total population in each district as the determiner of cluster size. As of the time of this writing, the "relaxation" portion of the algorithm has not been implemented. I plan to move onto different algorithms in the future.
  * Possible alternatives to SameSizeKMeans include: spectral clustering, and a from-scratch algorithm to cluster block-groups as nodes on a graph that I am currently in the midst of structuring in pseudo-code.
  * Ultimately, any algorithm must take a number of clusters as a *prespecified* input. The number of clusters is invariably the number of districts, and the number of districts is established by law.

Introduction to gerrymandering
------------------------------
Gerrymandering is the process of producing electoral districts such that the anticipated votes from these districts would disproportionately favor those who create them. A wonderful visualization of this process was produced by [The Washington Post](https://www.washingtonpost.com/news/wonk/wp/2015/03/01/this-is-the-best-explanation-of-gerrymandering-you-will-ever-see/?utm_term=.a0d638d12c92) as an adaptation of work by [Stephen Nass](https://www.reddit.com/r/woahdude/comments/2xgqss/this_is_how_gerrymandering_works/):

<img src='./images/gerrymandering_wp.png' style="width: 1000px">

This is the motivation this project: a method for producing districts in a prescribed, algorithmic manner that takes the the human element out of district creation. The hope is that this will lead to less gerrymandered, more representative districts.

Current districts
-----------------
### Congressional districts - 2016 election cycle
These districts were established as described [here](http://docs.legis.wisconsin.gov/statutes/statutes/3.pdf).
<img src='./images/congressional_districts.png' style="width: 1000px">
