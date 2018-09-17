## Introduction
### Ocean Circulation and Watermasses
Looking out at the ocean, it is hard to appreciate that the water lapping at your feet is actually on a roughly 1,000 year--somewhat circuitous--trajectory around the earth.  Our knowledge of ocean circulation and the geological, biological, and chemical processes that occur in the ocean come from the collection of measurements made at oceanographic and hydrographic stations of various chemical constituents (e.g. phosphate or oxygen, sometimes called "tracers" because they enable ocenographers to trace a process or trajectory) and properties (e.g temperature, salinity) at various depths ranging from the surface to the sea floor.  

Connecting similar values of a given property along strings of vertical profiles elucidates how water with similar properties (called  a watermass) flows, sinks, upwells, and mixes with other proximal watermasses. 

Consider sections along 25W and 175W.  The contours give a sense of the distribution of water of similar chemical composition. 


| Pacific | Atlantic|
|------------ | -------------|
![section of phosphate-Pacific](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_175-175w.png)|![section of phosphate-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_25-25w.png)|
![section of salinity-Pacific](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_175-175w.png)|![section of salinity-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_25-25w.png)|

### World Ocean Atlas
The World Ocean Atlas provides data on nitrate, phosphate, oxygen, temperature and salinity.  

## Part 1: Overview statistics

| Phosphate | Salinity|
|------------ | -------------|
![Phosphate-histogram](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/histograms/phosphate_alldata.png)|![Salinity-histogram](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/histograms/salinity_alldata.png)|

Looking at histograms of phosphate and salinity highlights how different tracers play different roles in the ocean and are influenced different processes.  The roughly normal distribution of salinity reflects that, while evaporation and precipitation, brine rejection and ice melt lead to areas of higher and lower salinity, the salinity of the majority of the ocean is unaffected by processes other than the slow mixing that occurs where watermasses meet.  Because salt is added and removed on very long timescales compared to that of ocean circulation, it salinity is an approximately conservative tracer. 

In contrast, phosphate is not conservative because there are highly active sources and sinks that add and remove it from the water.  Provided sufficient sunlight and complementary nutrients, organisms will incoprorate dissolved phosphate in tissue, reducing its concentration in the water column. When organisms die, they sink and many of them dissolve before reaching the seafloor, returning their phosphate deeper in the water column.  Not only does this process introduce phosphate to the internal ocean by a process other than watermass mixing, but the organic matter that reaches the seafloor may become buried, thereby removing that phosphate from the water column entirely. Sure enough, the phophate histogram has a right tail that suggests the highest phosphate water progressively mixes with lower phosphate water--similar to mixing seen in the mixing inferred from teh salinity histogram.  However, the left side of the distribution is not at all normal because, subject to light and other nutrient limiations, biology strips as much phosphate as from the water column.


## Part 2: Defining a watermasses and characteristic values
That said, on a more local scale, and particularly in the deeper ocean where the effects of biology are less pronounced, even phosphate can be used as a tracer. The signature values of deep watermasses are strongly influenced--dare I say, "dominated"--by the characteristic value of the source water and as result can often be used to trace the path and mixing of deep water through the ocean basins. 

Water of a given watermass is "formed" when it sinks or upwells to a new depth and takes on new physical and chemical properties. For example, the salinity of water traveling north as part of the gulf stream along the western edge of the North Atlantic increases due to evaporation, but when it reaches high latitudes, this warm surface water cools and becomes dense enough to sink to as deep as 3500m before starting its trip south as North Atlantic Deep Water (NADW). NADW is characteristically low nitrate and phosphate, high salinity and relatively warm--consistent with the characterisitics of its source water and relatively little mixing with another deep watermass at formation.  


### Q1: Is the deep water in the North Atlantic statistically different from the water in the deep South Atlantic?

![histograms and trajectories of tracers in the Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/tracer_hist_lonTraj_Atl.png)

Consider the above figure with tracer values at each of several depths along 25W between 68S and 55N for each of the five tracers in column A and histograms of these tracer values in column B. The figures in column A, With the exception of oxygen which is strongly affected by a number of processes, show values fairly consistent through the water column below 3000m at high latitudes, but show a departure in the mid latitutdes where the upper part of the water column (above 3500m) looks like the North Atlantic and the bottom part of the water column (below 4500m) looks like the Southern Ocean.  The trajectory for 4000m is pretty diagonal, connecting the high latitude values, suggesting that this is the depth (approximately) where mixing is happening at the margins between watermasses.

The two groups of tracer values apparent in the histograms in column B offer further support that the deep Southern Ocean water (Antarctic Bottom Water, AABW) and NADW are distinct watermasses.  Indeed when the tracer values in 50N and 70N and 40W and 20W were compared to the values in 50S and 70S and 40W and 20W at each depth, the two groups were distinct according to a t-test (p=.01) using all tracers.

### Q2: How far does Southern Ocean source water extend north?
In a simplified experiment, I considered whether deep water in the Atlantic Ocean to either most closely resemble a northern or southern end-member value.  I calculated the linear combination of the northern and southern sourced water (based on end-member values from that depth) at each point along 25degW, and colored the point based on which watermass representing a higher percentage of its makeup.  While this is a very rough approximation, it is possible to see some of the structure begin to emerge, including the Antarctic Bottom Water coming from the Southern Ocean and North Atlantic Deep Water filling the center of the Atlantic above it.
![two_endmember_analysis of Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/two_endmember_Atlantic.png)


### Q3: Can we trace water formation statistically?
Water formation/upwelling and stratification are both circulation patterns in which water moves from one place to another without considerable mixing, either along a vertical path or along a horizontal path, respectively. The below plots compare the level of connectedness as a function of the p-value calculated from two proximal water parcels 3deg x 3deg x 250m or 500m.  The higher the p-value (and thus the higher the likelihood that the values in these parcels were pulled from the same population), the darker the line, and higher the confidence in the connectedness.  

| Connectedness| Raw Data |
|------------ | -------------|
|         *Temperature*        ||
![temperature connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/temperature_connectedness_25W.png)|![section of temperature-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/temperature_section_80s-80n_25-25w.png)|
|         *Salinity*        ||
![salinity connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/salinity_connectedness_25W.png)|![section of salinity-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/salinity_section_80s-80n_25-25w.png)|
|         *Phosphate*        ||
![phosphate connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/phosphate_connectedness_25W.png)|![section of phosphate-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/phosphate_section_80s-80n_25-25w.png)|
|         *Oxygen*        ||
![oxygen connectedness in Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/connectedness/oxygen_connectedness_25W.png)|![section of oxygen-Atlantic](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/raw_demo_plots/oxygen_section_80s-80n_25-25w.png)|

## Part 3: What's next?  
### Multiple tracers and clustering
How can we use cluster analysis and multiple tracers to better define watermasses?  What areas are classified differently depending on the combination of tracers used? Are those differences interpretable?

### Endmember analysis to trace flow trajectory
Extending the endmember anlaysis to trace trajectories, the next level analysis is to acknowledge that the mix varies with longitude, and to analyze at various longitudes, tracing the distribution of the highest percentage of southern ocean source water as it moves north see the extent to which it fills the abyssal ocean.  Higher up in the water column tracing a high and potentially characteristic fraction southern ocean source water might yield the pathway the water takes as it snakes its way from the Southern Ocean to the North Atlantic.  

### Role of non-mixing processes on tracer distribution
To understand the contribution by a given process, we need to know how much change is due to mixing; if we use a one dimensional mixing model for salinity and then repeat the process for phosphate, how different are the mixtures? statistically different?


