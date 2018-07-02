
## Testing methods for sectioning the water column with statistics
One fo the next questions in my mind was: how might I go about dividing the water column into distinct zones with math. 

Figure | Caption
------------ | -------------
![profile of nitrate](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/watermass_diff_plts/nitrate_plan_60:50n-40:0w_and_50:60s-40:0w_4000m_ptest.png) | Comparison of two areas of the North Atlantic (50:60N, 40:0W) and Southern Ocean (50:60S, 40:0W) using a ptest to demonstrate they are statistically different

### Pseudo code:
Find depths of statistically different water masses by 1 tracer, assumes miniumum sample size of 200 of both zones

- Starts from the bottom and if the samples between bottom bound and middle bound are not statistically different from the sample between middle bound and upper bound, push the middle bound and upper bound up and try again. 
- Once they are stastistically different, drop the middle bound incrementally until reach the depth where the p value is just less than .01, then set bottom bound = middle bound and repeat

Figure | Caption
------------ | -------------
![section and column of nitrate](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/watermass_diff_plts/nitrate_section_70-0s32w_columnwptest_40-30s30-25w.png)| Section view of nitrate (70:0S, 32W) with dotted line location of column location and accompanying column plot (30:40S, 25:30W) broken into potential watermasses with ptest


### Pseudo code:

Break water column into statistically different watermasses by regressing sections of the slope of at least 60 data points. 
- As long as the r2 is increasing, expand the intervale by 5 meters. 
- When the r2 value begins to decline save the depth and start a new section

Figure | Caption
------------ | -------------
![section and column of nitrate](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/watermass_diff_plts/nitrate_section_70-0s32w_columnwslope_40s30w.png)| Section view of nitrate (70:0S, 32W) with dotted line location of column location and accompanying column plot (30:40S, 25:30W) broken into potential watermasses with r**2 analysis of slope at different locations on column profile
