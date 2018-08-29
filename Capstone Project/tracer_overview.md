## Introduction
### Ocean Circulation and Watermasses
Looking out at the ocean, it is hard to appreciate that the water lapping at your feet is actually on a roughly 5,000 year--somewhat circuitous--trajectory around the earth.  Our knowledge of ocean circulation and the geological, biological, and chemical processes that occur in the ocean come from the collection of measurements made at oceanographic and hydrographic stations of various chemical constituents (e.g. phosphate or oxygen, sometimes called "tracers" because they enable ocenographers to trace a process or trajectory) and properties (e.g temperature, salinity) at various depths ranging from the surface to the sea floor.  

Connecting similar values of a given property along strings of vertical profiles elucidates how water with similar properties (called  a watermass) flows, sinks, upwells, and mixes with other proximal watermasses. 

Consider sections along 25W and 175W.  The contours give a sense of the distribution of water of similar chemical composition. 


| Pacific | Atlantic|
|------------ | -------------|
![section of phosphate-Pacific](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_175-175w.png)|![section of phosphate-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_25-25w.png)|
![section of salinity-Pacific](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_175-175w.png)|![section of salinity-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_25-25w.png)|

### World Ocean Atlas
The World Ocean Atlas provides data on nitrate, phosphate, oxygen, temperature and salinity.  

## Part 1: Overview statistics
Looking at histograms of phosphate and salinity highlights how different tracers play different roles in the ocean and are influenced different processes.  The roughly normal distribution of salinity reflects that, while evaporation and precipitation, brine rejection and ice melt lead to areas of higher and lower salinity, the salinity of the majority of the ocean is unaffected by processes other than the slow mixing that occurs where watermasses meet.  Because salt is added and removed on very long timescales compared to that of ocean circulation, it salinity is an approximately conservative tracer. 

In contrast, phosphate is not conservative because there are highly active sources and sinks that add and remove it from the water.  Provided sufficient sunlight and complementary nutrients, organisms will incoprorate dissolved phosphate in tissue, reducing its concentration in the water column. When organisms die, they sink and many of them dissolve before reaching the seafloor, returning their phosphate deeper in the water column.  Not only does this process introduce phosphate to the internal ocean by a process other than watermass mixing, but the organic matter that reaches the seafloor may become buried, thereby removing that phosphate from the water column entirely. Sure enough, the phophate histogram has a right tail that suggests the highest phosphate water progressively mixes with lower phosphate water--similar to mixing seen in the mixing inferred from teh salinity histogram.  However, the left side of the distribution is not at all normal because, subject to light and other nutrient limiations, biology strips as much phosphate as from the water column.


## Part 2: Defining a watermasses and characteristic values
That said, on a more local scale, and particularly in the deeper ocean where the effects of biology are less pronounced, even phosphate can be used as a tracer. The signature values of deep watermasses are strongly influenced--dare I say, "dominated"--by the characteristic value of the source water and as result can often be used to trace the path and mixing of deep water through the ocean basins. 

Water of a given watermass is "formed" when it sinks or upwells to a new depth and takes on new physical and chemical properties. For example, the salinity of water traveling north as part of the gulf stream along the western edge of the North Atlantic increases due to evaporation, but when it reaches high latitudes, this warm surface water cools and becomes dense enough to sink to as deep as 3500m before starting its trip south as North Atlantic Deep Water (NADW). NADW is characteristically low nitrate and phosphate, high salinity and relatively warm--consistent with the characterisitics of its source water and relatively little mixing with another deep watermass at formation.  


### Q1: Is the deep water in the North Atlantic statistically different from the water in the deep South Atlantic?

![histograms and trajectories of tracers in the Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/tracer_hist_lonTraj_Atl.png)

Consider the above figure with tracer values at each of several depths along 25W between 68S and 55N for each of the five tracers in column A and histograms of these tracer values in column B. The figures in column A, With the exception of oxygen which is strongly affected by a number of processes, show values fairly consistent through the water column below 3000m at high latitudes, but show a departure in the mid latitutdes where the upper part of the water column (above 3500m) looks like the North Atlantic and the bottom part of the water column (below 4500m) looks like the Southern Ocean.  The trajectory for 4000m is pretty diagonal, connecting the high latitude values, suggesting that this is the depth (approximately) where mixing is happening at the margins between watermasses.

The two groups of tracer values apparent in the histograms in column B offer further support that the deep Southern Ocean water (Antarctic Bottom Water, AABW) and NADW are distinct watermasses.  Indeed when the tracer values in 50N and 70N and 40W and 20W were compared to the values in 50S and 70S and 40W and 20W at each depth, the two groups were distinct according to a t-test (p=.01) using all tracers.

### Q2: Can we trace water formation statistically?
We can 

| Raw Data | Connectedness|
|------------ | -------------|
![temperature connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/temperature_connectedness_25W.png)|![section of temperature-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/temperature_section_80s-80n_25-25w.png)|
  :-----------: | -----------: |
          *Salinity*        ||
![salinity connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/salinity_connectedness_25W.png)|![section of salinity-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_25-25w.png)|
![phosphate connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/phosphate_connectedness_25W.png)|![section of phosphate-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_25-25w.png)|
![oxygen connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/oxygen_connectedness_25W.png)|![section of oxygen-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/oxygen_section_80s-80n_25-25w.png)|

## Part 3: What's next?  
### Multiple tracers and clustering
How can we use cluster analysis and multiple tracers to better define watermasses?  What areas are classified differently depending on the combination of tracers used?




- To understand the contribution by a given process, we need to know how much change is due to mixing

	- if we use a one dimensional mixing model for salinity and then repeat the process for phosphate, how different are the mixtures? statistically different?

- how do you mathematically define a watermass?
	- seperate out by derivative analysis (strong slope indicates two watermasses passing with minimal mixing, weak slope indicates similar watermasses (check another tracer) or mixing)
	- verify layers are distinct by p-test 
- how do you establish its characteristic value?
	- find the value at the water formation site (where is the water formed?)
		when the combination of characteristic watermasses from surface to depth 
	- check that there is a continuous trajectory over which that value persists (ptest?)
	- when the value ceases to persist check to see if there is another watermass present

In order to get around the proximity to land end-member problem, I attempted to verify that there is no coastline closer than 2 degrees away





## Graveyard:
Different tracers are sensitive to different processes. For example, salinity goes up when water evaporates and down when there is precipitation or ice melt, but is mininmally affected by organisms.  The concentrations of nitrate and phosphate, on the other hand, go down when organisms consume them, but go up again when the organisms die and dissolve. 

Chemical Oceanography
Very briefly, it's worth explaining that chemical oceanography is the study of the chemical constituents of the ocean--everything from complex proteins to individual ions.  By studying the concentrations of these constituents both temporally and spatially, we can gain insight into the biological, geological, chemical, and physical processes at work in the ocean. 






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
