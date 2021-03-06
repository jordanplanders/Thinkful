# Unit 1
## Lesson 1
### Assignment 1: Data, Engineering and Machine Learning
- Technology stack- collection of elements that make up a product
	- front end - interface that users interact with
	- back end - servers, services, databases etc; heavy lifting that feeds data to the front end
- Related roles
	- Data Analyst: draw conclusions and generate reports (proto data scientist)
	- Data Engineer: gather and store data, create and manage data pipelines, databases; less interpretation of data
	- Machine Learning Engineer: algorithms and modeling kwth an emphasis on algorithm design and efficiency
	- Data Scientist: whatever the recruiter says it is
### Assignment 2: The data science toolkit
- Python
- Packages
	- numpy
	- pandas
	- matplotlib
	- seaborn
	- scikit-learn
	- StatsModels
- SQL- Structured Query Langauge
	- access and preprocess data
### Assignment 3: Thinking like a data scientist
- curiosity, practicality (have to define questions with limited scope that can be answerd with data), skepticism (how confident are we in the results we see; lies, damn lies, and statistics)
- sometimes the way an abstract question is translated into a concrete one is dictated by the available data
- findin and evaluating data sources: data archives, repositories, web scraping, logs, documents...
- evaluating uncertainty: assess how certain we are that conclusions based on a particular statistic are valid; are there flaws in the source of the sample (sampling method, representative nature of the sample for the population etc), size of the sample, noise (variance) in the data 
### Assignment 4: Drill: What can data science do?
Take the following scenarios and describe how you would make it testable and translate it from a general question into something statistically rigorous
1. You work at an e-commerce company that sells three goods: widgets, doodads, and fizzbangs. The head of advertising asks you which they should feature in their new advertising campaign. You have data on individual visitors' sessions (activity on a website, pageviews, and purchases), as well as whether or not those users converted from an advertisement for that session. You also have the cost and price information for the goods.
		- a person who saw an ad has four outcomes: buy nothing, buy w, buy d, or buy f
		- we want to know which ad-buy pair results in the highest sales and whether that number is statistically significant
		- Not clear what a follow up test would be given that it appears all the combinations have already been tested...
		
2. You work at a web design company that offers to build websites for clients. Signups have slowed, and you are tasked with finding out why. The onboarding funnel has three steps: email and password signup, plan choice, and payment. On a user level you have information on what steps they have completed as well as timestamps for all of those events for the past 3 years. You also have information on marketing spend on a weekly level.
		- Did sign-ups slow because fewer people started the sign-up process or because more people fell out of the pipeline on the way to completion? What ad campaign was on when the person started the process? 
		- find the time when the max number of people got through step 3, calculate whether that is statistically different from the last several weeks.  Then look at the relationship between the number of people who got through step 1 and 2 for both timeframes to see where people fell out of the pipeline.  Then consider the marketting strategy during that time (assuming they are statistically different) and possibly reimplement that approach for a few weeks, track the numbers and compare against current status. 

3. You work at a hotel website and currently the website ranks search results by price. For simplicity's sake, let's say it's a website for one city with 100 hotels. You are tasked with proposing a better ranking system. You have session information, price information for the hotels, and whether each hotel is currently available.

4. You work at a social network, and the management is worried about churn (users stopping using the product). You are tasked with finding out if their churn is atypical. You have three years of data for users with an entry for every time they've logged in, including the timestamp and length of session.
		- examine the average length of time between sign up and last login of users grouped by sign-up week

### Assignment 5: Challenge: Personal goals
- academic style problems, but also some of these churn questions (I reckon)
- wrangling big datasets (dip into spark), wrangling small datasets (what constitutes over-mining data)
- hypothesis testing
- signal processing
## Lesson 2: SQL: data access methods
### Assignment 1: Introduction to Databases
- Database contexts
	- operational layer- part of the application responsible for delivering the core user experience; server and client-side application code
	- storate layer- database used to store information rather than storing locally in text files of some kind
		- databases can be distributed across many machines and have higher capacity 
		- multiple users can access a remote database at the same time
		- databases can take data from multiple sources
	- analytics layer
- database structure
	- relational databases consist of a series of tables, with each table having its own defined schema and a number of records (rows)
		- each table has rows and columns; each column has a name and a data type associated with it
		- database schema is a particular configuration of tables with columns
			- schema can change, but migrating (systematic update) data to new structure is costly (tricky to make sure that things won't break in the move and that no data is lost)
		- each record (row) must conform to the schema and ideally will have a value in every field (column) for efficiency (though realistically thre are NULL, blanks and N/As in tables)
	- three kinds of tables:
		- raw- contain simple, relatively unprocessed data (closely resembles the data produced by the operational services)
		- processed tables- contain data that's been cleaned and transformed to be more readable/useable
		- roll-up tables- specific kidn of processed table that take data and aggregate it @TODO
- SQL- Structured Query Language
	- language used to create, retrieve, update and delete database records
	- there are several flavors of SQL with meaningful minor differences; PSQL, SQLite, MySQL
	- good to remember that SQL sees the world in rows, and we'll want to think about attributes at a rows level (with some grouping and aggregating)
### Assignment 2: Setting up PSQL (already done)
- psql -d database_name -f file.sql
### Project 3: SQL Basics
- The CREATE clause - to create a new table 
		- lowercase for table names, no spaces (underscores)
		- each column has a column name and a TYPE (example below has arbitrary types filled in), but can also have contraints like prohibiting null values; column creation lines are seperated by commas
		CREATE_TABLE table_name (
			some_column_name FLOAT column_constraint,
			some_second_column_name TEXT, 
			...
			last_column_name TYPE);
- The SELECT and FROM clause
	- SELECT retrieves rows FROM a table
		- select all rows:
			SELECT * FROM someTable;
		- select specific rows
			SELECT some_column_name, last_column_name from table_name; 
- aliasing
	- can SELECT a column and return it with a different label; useful when selecting across multiple tables that reuse column names
			SELECT some_column_name AS col1 FROM table_name;
- Filtering with WHERE
	- allows us to specify a set of conditions that the results must meet
		- LIKE - for pattern matching when analyzing string data
		- BETWEEN - check if a value is between a pair of values
		- AND and OR - linking conditions together
- Ordering with ORDER BY
	- control the order in which results are returned 
	- can link together multiple ordering conditions 
			ORDER BY some_column_name DESC; 
- Limiting with LIMIT
	- limit the number of results returned, for example to get the top 5
- Formatting notes:
	- new lines for everything, indent column specific instructions
	- ALL CAPS for SQL instructions, whatever casing is relevant for other names
#### Exercises:
 https://gist.github.com/jordanplanders/298877231bb950a192223c681754dd56
	- - 1. The IDs and durations for all trips of duration greater than 500, ordered by duration.
	SELECT 
		trip_id,
		duration 
	FROM 
		trips 
	WHERE
		duration >500
	ORDER BY
		duration;
	  
	- - 2. Every column of the stations table for station id 84.
	SELECT *
	FROM 
		stations
	WHERE 
		station_id = 84;
	  
	- - 3. The min temperatures of all the occurrences of rain in zip 94301.
	SELECT 
		mintemperaturef
	FROM 
		weather
	WHERE 
		events = 'Rain'
		AND zip = 94301;
### Project 4: Aggregating and grouping
- GROUP BY- 
	- comes after WHERE clause and before ORDER BY clauses;
	- without aggregating function, just gets rid of duplicate entries
	- all columns in GROUP BY clause must also be in SELECT statement
	- can use col numbers instead of col names
- Aggregators
	- functions that take a collection of values and return a single value
	- return a column labelled by function, not column so need to alias 
	- AVG, MIN, MAX, COUNT(*) 
#### Exercises: https://gist.github.com/jordanplanders/1c3e994b6246aa51cdba8b4e65166f28
	-- 1. What was the hottest day in our data set? Where was that?
	SELECT 
	  maxtemperaturef, 
	  zip 
	FROM 
	  weather 
	ORDER BY 
	  maxtemperaturef DESC
	LIMIT 1;
	
	-- 2. How many trips started at each station?
	SELECT 
	  COUNT(trip_id), 
	  start_station 
	FROM 
	  trips 
	GROUP BY 
	  start_station
	ORDER BY 
	  COUNT(trip_id) DESC;
	  
	-- 3. What's the shortest trip that happened?
	SELECT * 
	FROM 
	  trips 
	ORDER BY 
	  duration 
	LIMIT 1;
	
	-- 4. What is the average trip duration, by end station?
	SELECT 
	  end_station, 
	  avg(duration) 
	FROM 
	  trips
	GROUP BY 
	  end_station 
	ORDER BY 
	  avg(duration);
### Project 5: Joins and CTEs
- Basic Joins
	- in a join clause indicate one or more pairs of columns you want to join the two tables on
	- by default SQL performs an inner join (only returns rows that are succesfully joined from the two tables)
	- comes after the FROM statement (if multiple tables, order matters), followed by an ON clause that specifies the table.columns that should be the same (to link the two)
	
		SELECT
			table1.col1.
			table1.cal2
			table2.col3
			table2.col4
		FROM 
			table1
		JOIN
			table2
		ON	
			table1.col2 = table2.col2
- Table aliases
	- in a join, often useful to alias table to simplify table names or add a table name (in the case of a self join) in the ON clause
- Types of Joins
	- (INNER) JOIN: only returns rows that are successfully joined
	- LEFT (OUTER) JOIN: returns all rows from the left table even if no common rows in right table; rows without a match wil be filled with NULL
	- RIGHT (OUTER) JOIN: same as a LEFT (OUTER) JOIN if you reverse the table order in the FROM and JOIN clauses
	- (FULL) OUTER JOIN: returns all rows with NULLs in all places where the join doesn't fill in data
- CTEs (Common Table Expressions)
	- since a join statement returns a table, can join a join statment to other tables/the results of other queries
	- note: JOINs happen before aggregate functions so if you want aggregate information about one table and information from the other table, but you don't want information from the other table to weight the results of the aggregate, create the first table with aggregation, THEN join to the second table
		- if you join station with trips and calculated average lat and lon of all start_stations in a city, it will be the average location of start_station in a city of all trips, however, if calculate average start_station from station table and then join, it will be the average start_station location for the set of unique stations in a city.

		WITH intermediate_table_name AS (query1)
	- multiple joins are common to collect information from multiple tables
- Case
	- set up conditions then take action in a column based on them
	- CASE WHEN condition THEN value ELSE value END
	- CASE statemnts go in the SELECT column and indicate what value to return given a conditional statment then aliased 
#### Exercises:
 https://gist.github.com/jordanplanders/f5d960fa280c27d5772a93b6bd268bf0
 -- 1. What are the three longest trips on rainy days?  
SELECT 
	trips.trip_id,
	weather.date,
	trips.duration, 
	weather.events  
FROM 
	trips 
JOIN 
	weather
ON 
	weather.date = SUBSTRING (trips.start_date ,0 , 11 ) 
WHERE 
	weather.events = 'Rain' 
GROUP BY 
	trips.trip_id,
	weather.date, 
	trips.duration,
	weather.events
ORDER BY 
	weather.date 
LIMIT 300;

-- 2. Which station is full most often?
--FIND WHICH STATION IS FULL (DOCKS_AVAILABLE = BIKES_AVAILABLE) MOST OFTEN (NUMBER OF STATUS UPDATES WHERE THIS WAS TRUE)

WITH station_full
AS(
	SELECT 
		station_id, 
		COUNT(station_id) as times 
	FROM 
		status 
	WHERE 
		status.bikes_available = status.docks_available 
	GROUP BY 
		station_id 
	ORDER BY 
		COUNT(station_id) DESC
	LIMIT 1)
	
-- MATCHING A STATION NAME WITH THE STATION_ID FROM ABOVE
SELECT 
	stations.name, 
	station_full.times 
FROM 
	stations 
JOIN 
	station_full 
ON 
	stations.station_id = station_full.station_id; 
	
-- 3. Return a list of stations with a count of number of trips starting at that station but ordered by dock count.
-- Query to find number of trips started at each station
WITH 
	stations2 AS 
	(
		SELECT 
			stations.station_id AS station_id, 
			stations.name, COUNT(*) as trips_started
		FROM 
			stations 
		JOIN 
			trips 
		ON 
			stations.name = trips.start_station 
		GROUP BY 
			trips.start_station, stations.station_id, stations.name)
			
-- QUERY TO ORDER STATIONS2 BY NUMBER OF DOCKS AVAILABLE
SELECT 
	status.docks_available, 
	stations2.name, 
	stations2.trips_started 
FROM 
	stations2 
JOIN 
	status
ON 
	stations2.station_id = status.station_id
ORDER BY 
	status.docks_available DESC
LIMIT 10;

-- 4. (Challenge) What's the length of the longest trip for each day it rains anywhere?
-- QUERY TO ONLY TRIPS WHEN IT'S RAINING
WITH raining AS (
	SELECT 
		weather.date AS date, 
		trips.start_date AS tmestamp, 
		trips.duration AS duration 
	FROM 
		weather 
	JOIN 
		trips 
	ON 
		weather.date = SUBSTRING(trips.start_date, 0, 11) 
	WHERE 
		events = 'Rain')


SELECT 
	max(duration)/360 AS max_duration_hrs, 
	date AS start_date 
FROM 
	raining 
GROUP BY 
	date 
ORDER BY 
	date 
LIMIT 100;
### Project 6: Airbnb Cities
- What's the most expensive listing? What else can you tell me about the listing?  
		
		WITH max_price_list AS (
			SELECT * from calendar 
			WHERE cast(price as float) >0 
			ORDER BY cast(price AS float) DESC 
			LIMIT 1)
			
		SELECT 
			listings.neighbourhood, 
			listings.room_type, 
			listings.minimum_nights, 
			max_price_list.date, 
			max_price_list.price 
		FROM listings 
		JOIN max_price_list 
		ON cast(max_price_list.listing_id AS int) = listings.id;

		Calabasas	Entire home/apt	2	2018-12-22	61000.00
- What neighborhoods seem to be the most popular?  
		
		WITH taken_list as (
			SELECT 
				count(*) AS num_taken_list, 
				listing_id 
			FROM calendar 
			WHERE available = 't' 
			AND substring(date, 0, 5)= '2018'
			GROUP BY listing_id )
				
		SELECT 
			listings.neighbourhood, 
			SUM(taken_list.num_taken_list) AS num_taken 
		FROM listings 
		JOIN taken_list 
		ON cast(taken_list.listing_id as int) = listings.id 
		GROUP BY listings.neighbourhood
		ORDER BY SUM(taken_list.num_taken_list) DESC
		LIMIT 5; 

		neighborhood	num_taken_nights
		Hollywood	181607
		Venice	158504
		Downtown	107685
		Long Beach	86799
		Santa Monica	67395

- What time of year is the cheapest time to go to your city? November  
		SELECT 
			AVG(cast(price AS float)),
			substring(date, 6, 2) AS mo  
		FROM calendar  
		GROUP BY substring(date, 6, 2);
		
		average price		month
		231.06087495398208	11
		
- What about the busiest? November  
		WITH freerooms AS(
			SELECT 
				COUNT(*) AS free, 
				substring(date, 6, 2) AS mo 
			FROM calendar 
			WHERE available = 't' 
			GROUP BY substring(date, 6, 2)),

		busyrooms AS(
			SELECT 
				COUNT(*) AS busy, 
				substring(date, 6, 2) AS mo 
			FROM calendar 
			WHERE available = 'f' 
			GROUP BY substring(date, 6, 2))
		
		SELECT 
			cast(busyrooms.busy as float)/(cast(freerooms.free as float) +cast(busyrooms.busy as float)) AS busy_room_rate, 
			busyrooms.mo 
		FROM busyrooms 
		JOIN freerooms 
		ON busyrooms.mo = freerooms.mo;
## Lesson 3: Intermediate visualization
### Assignment 1: The basics of plotting review
- Basic plot types
	- line plots- data over some continuous variable
	- scatter plots- relationship between two variables
	- histograms- distribution of a continuous dataset
	- bar plot- counts of categorical variables
	- QQ plot- how close a variable is to a known distribution & outliers
	- box plot- compare groups and identify differences in variance & outliers
### Assignment 2: Formatting, subplots, and seaborn
- seaborn @TODO let's talk about the structure of the seaborn package
	- sns.load_dataset() @TODO what form does the data need to be in for it to load properly?
	- sns.set(style = )
	- sns.despine()
	- sns.FacetGrid()
		- .map(plottype, variable)_to_be_plotted)
	- sns.boxplot(x = , y = , hue = , data = )
	- sns.factorplot(x= , y= , hue= , data= , kind=* , ci=, join= , dodge= )
		- bar: bar plot
		- point: like a bar plot but more efficient; good for points that have error bars; may or may not be connected
	- sns.lmplot(x= , y= , hue= , data= , fit_reg= , ci = , lowess =  scatter_kws={})
		- scatter plot 
			- fit_reg: with/without a regression line 
			- ci: with/without confidence interval error cloud @TODO I don't know how to calculate  this error cloud manually
			- lowess: using local wighing to fit a line @TODO
			- col= " parameter results in the data being split out by category and plotted in subplots @TODO
#### Drill 3: Presenting the same data multiple ways
https://github.com/jordanplanders/Thinkful/blob/master/Bootcamp/Unit%201/bike_data/Kevin_bike_datavis.ipynb
### Assigment 4: Cleaning Data
- Finding Dirt
	- anomolous values
	- fake answers (straightlining, repeating answer sequence, time to finish below some threshold)
- Cleaning Dirt
	- replace with NULL or None
	- map to a valid response (an extreme value maps to the highest non-outlier response, winsorizing )
	- remove (duplicate entries)
	- other (data issues that are systemic, widspread, or for a particuar data-related reason)

	- clean with code so that there's a record of the cleaning process
	- don't alter original data, keep a separate "clean copy"
### Assignment 5: Manipulating strings in Python
- string methods
- re (regular expresssions or regex)
	- a regular expression is a sequence of characters that defines a search patterns
	- not always more efficient than string methods
- extracting different categories of character from a string
	- isdigit()
	- isalpha()
	- isnumeric()
	- isspace()
	- isalnum()
- Apply: .apply() allows one to apply a method to each element in a data frame or series
- lambda functions: small, temporary, unnamed function of the format:
			lambda x: f(x) (if [condition] else [alternative])
		- one line functions that would usually be:
			def function(x):
				return f(x)
- filter: 
		- returns an iterator of booleans (based on a function that returns booleans) that when applied to a series or a string only returns entries/characters that are True. @TODO @WTF
			- list(filter(lambda x: boolean function, series))
			- ex: list(filter(lambda x: str(x).isdigit(), money))
			- series.apply(lambda x: ''.join(list(filter(boolean function, string)))
			- ex: money.apply(lambda x: ''.join(list(filter(str.isdigit, str(x)))))
- splitting strings apart: split a string into a list of strings; by default splits at spaces, but can be split at some other character or string
	- pandas has its own version: series.str.split(delimeter, expand = True) that will split the series of strings on the delimeter or regex pattern and return a set of series that correspond to the first, second, third, ... nth elment in the split
		- ex: word_split = words.str.split('$', expand=True)
				names = word_split[0]
				emails = word_split[1]
- replace: replace specific characters or strings with a new string
	- pandas has its own version: series.str.replace(str1, str2)
- changing case: often it will be useful to standardize the case either with: .lower(), .upper(), .capitalize
- stripping whitespace: string.strip(), also lstrip(), rstrip()
	- pandas has its own version for whitespace: series.str.strip()
### Assignment 6: Exercise on cleaning data
### Assignment 7: Missing data
- missing data can be systematicall missing, which raises questions about dataset reliability
- even if missingness is random, analysis can break because of missing values --> df.dropna() is a built in method in pandas to drop all rows with a missing value
- When does missingness matter?
	- so many rows have to be thrown out that the set loses statistical power
	- systematic missing values causes systematic subsets of data to be thrown out making the dataset biased

	- MCAR- missing completely at random: three year old inserts crayons into a random server in a server room and corrupts a drive (could have been a three year old from anywhere given its random that one would be there in the first place, and they picked a random server and a random place to put crayons)
	- MAR- missing at random: if a particular group is likely to skip a question regardless of response and we know this, we can explain the absense of the data and carefully  work around it
		- check correlation between missing scores and various variables to identify what is lurking
	- MNAR- missing not at ransom: if samples that are likely to have a particular value are absent, stop
- Imputing Data- guessing at what values would fill empty fields
	- replacing with the mode, median, or mean works for keeping central tendancy the same, but reduces the variance and alters correlations with other variables
	- can group existing entries into similar groups and impute strategically if data
- Beyond Imputation
	- sometimes its possible to collect more data either in a focused way targetting the MAR group or not if its an MCAR problem
## Lesson 4: Experimental Design
### Project 1: A/B tests
- One of many possible experiemental designs used to identify whether one version of an object of interst is better at producing a deisred outcome
- Components: 
	- Two versions of something whose effects will be compared (preferably a control version and an alternate, though realistically it is sometime two unknowns)
	- a sample (representative of the population), divided into two groups (that are the same in composition and preferably randomly chosen)
	- a hypothesis stating an expectation of what will happen
	- identified outcome(s) of interest; the measurable key metric that will be used to identify/characterize change
	- other measured variables: measure the hell out of everything to help check that the two groups were sufficiently similar, and to idnetify other responses to the change 
- Getting a good sample: the sample has to be representative of the population and any differences in outcome should be due to differences in treatment
	- easy when there's a constant flow and it's possible to just sample the flow
	- hard when it has to be all or nothing (music on, music off)
- Key to key metrics
	- metric as close to the business goal as possible; a metric that reflects an intermediate step and doesn't measure the final outcome doesn't really help
	- metrics that are reliably measurable, preferably somethign passively measured and not based on specific engagement with subjects or self-reported data
	- metrics may have different time windows; it may take a few months for something to surface as a win or a loss.
#### Exercises 
	- Does a new supplement help people sleep better?
		Hypothesis: would presumably be that the supplement would help people sleep better
		Sample: It's difficult to sample the population at large, but perhaps by getting people to opt in via their PCPs, making sure that the percentage of people who say they don't sleep well matches the national average would be a start.  
		Experiement: The experiment would involve sleep studies of all people without the supplement as a baseline, measuring reported quality of sleep as well as EEG and other biometrics.  Then the group would be split and half the sample would be given a placebo adn half the group would be given the supplement and the sleep study would be repeated.  Ideally there would be more than one night on each side, but realistically it would be two nights per subject. 
		Key Metric: fewer bouts of restlessness, less EEG activity, or self reported ratings.  (As a lay person in the field, I would probably have to identify the key metric from the baseline night reports)

	- Will new uniforms help a gym's business?
		Hypothesis: new uniforms will help a gym's business.  
		Sample: People who walk into a gym  
		Experiemnt: This is an all-on, all-off scenario, but potentilly by taking September as a control month and measuring October  with treatment applied, one could avoid seasonal effects of upcoming holidays, or post-holiday season, or pre-summer, or summer vacation lulls.  
		Key Metric: "Help" would likely be based on revenue per month, or perhaps revenue per quarter if there are discounts on new sign ups that would need take time to  manifest.
		Additional measured variables: new information seekers, new sign ups, number of services added to existing subscriptions, number of people re-upping their memberships, potentially surveying about quality of service/professionalism as a back of the envelope impression
		
	- Will a new homepage improve my online exotic pet rental business?
		Hypothesis: a new homepage will help with exotic pet rental business
		Sample: visitors to exotic pet rental site
		Experiment: Use a splitting server to show some people the new site and some people the old site and track the outcomes (e.g. split.io)
		Key Metric: increase in monthly revenue?
		Additional: number of new rentals, number of extended rentals, differences in the types pets rented
		
	- If I put 'please read' in the email subject will more people read my emails?
		Hypothesis: adding 'please read' in email subject lines will prompt more people to read emails
		Sample: listserve
		Experiment: Create a sample set from the email list and send half the altered subject line, and half the standard subject line for the same email with the same send date/time. 
		Key Metric: "people who read emails" as those who clicked on something in the email or followed up
		Additional variables: open rate, unsubscribe rate
### Assignment 2: Simpson's paradox
- phenonmenon in which the average over a number of groups shows one trend, but the average for each individual group shows the opposite or no trend (luriking variable paradox: an unaccounted for varaible changes the relationship between two other variables)
- using randomization to make sure splits don't have lurking demographic tendancies can help
	- confirm the groups are similar before interpreting your results.  
	- Make it a habit to look at subgroups within your A/B test to make sure the overall trend is reflected in the subgroups.  
	- If the subgroups differ from the overall trend, your question should guide whether you report conclusions based on the overall sample, the subgroups, or both.  You don't want to advocate for condition A, even if it performs better overall, if condition B actually works better within every subgroup
### Project 3: Bias and A/A Testing
- bias- anything that causes a sample to systematically differ from the population
	- sampling bias/selection bias: when the sample differs from the population in a systematic way
	- assignment bias: when the sample is split in a way that makes the make-up of the groups to differ
	- contextual bias: when a feature of the environment of testing prompts people in one group to have a different experience (and thus to potentially feel differently about the situation than they would otherwise)
	- observer bias: when the tester/interviewer interferes with the testing (which is to say, interacts with the subjects substantially) 
- A/A Testing- comparing the out come of choice between two identical versions of something.  Sets a baseline for what the difference might be between groups even in the situation in which nothting was different
	- testing method errors can be exposed
	- sample split errors can be exposed
	- sample size errors can be exposed (perhaps the event is to rare to detect in the planned sample size)
#### Drill: Am I biased?
- You're testing advertising emails for a bathing suit company and you test one version of the email in February and the other in May.
	- The design of the study does not appear to take into account the seasonality and effect of geography on bathing suit sales. Are the subjects in the northern or southern hemisphere?  Are they from a cold place where people are more likely to go on holiday to warm places in March, or is summer the only bathing suit season? Anyway around it, there are contextual biases lurking. 
- You open a clinic to treat anxiety and find that the people who visit show a higher rate of anxiety than the general population.
	- Primarily people concerned about anxiety will visit the clinic so the visiting population won't reflect the composition of the population at large leading to selection bias.
- You launch a new ad billboard based campaign and see an increase in website visits in the first week.
	- A billboard campaign will disproportionately affect people local to a particular region, and/or people in cars passing by. These two groups may or not be representative of the population at large 
- You launch a loyalty program but see no change in visits in the first week.
	- A week is likely too short a window for measuring shifts in behavior prompted by a program that likely involves accrueing points. Without more information about the program, unless people were automatically enrolled and there was a huge marketting push around awareness, it is unlikely to sway behavior immediately (particularly if users have to opt-in explicitly).
### Project 1-4-4
https://docs.google.com/document/d/1RK7Uil3IxYxlqCezrhFsfb0VaHFaqWGoQR3kAjMK9vk/edit?usp=sharing
### Project 1-4-5: The research proposal
- The problem: 
	- define the question or problem
	- justify why the problem should be studied
	- review what we already know about the problem
- The potential solution
	- a potential solution is also a hypothesis about what might solve the problem
- The method of testing the solution
	- design of the experiement
	- analysis plan
	- benchmarks (key metrics, points of interest)
- Why bother?
	- adjust disconnect between question and study design
	- adjust study design that will not generate usable data
	- account for false positives
	- prevents mixed expectations about what will be done and how it will be executed
#### Drill: 
Prompt: 
To prevent cheating, a teacher writes three versions of a test. She stacks the three versions together, first all copies of Version A, then all copies of Version B, then all copies of Version C. As students arrive for the exam, each student takes a test. When grading the test, the teacher finds that students who took Version B scored higher than students who took either Version A or Version C. She concludes from this that Version B is easier, and discards it.

Plan:
Problem: Students cheat on exams.  By using three versions of an exam, it is more difficult for students to look at each others' paper for answers because the papers are not necessarily the same type. However, but administering three versions of the test, it is also possible that one of those tests will be notably easier or harder than the other two, causing the grades to be uncomparable.  

Potential Solution: Using multiple exam versions is a reasonable strategy for combatting this problem, however it should be executed and calibrated as carefully as possible.  Exams should be collated ABC and passed out to students only once all are present and seated (preventing potential clusters of A exams, for example).

Experiment: Once the test has been administered, results should be examined for student subgroups. Did one test have notable higher or lower scores than the other two? (ANOVA?)  Did students deviate notably and inexplicably from their historical performance? (paired t-tests between past exam scores and current scores?) Making sure each test taker was surrounded by students taking different versions by passing out the exam carefully, should significantly reduce the probability of cheating occuring on a particular version.  


### Assignment 1-4-6: AB Testing and t-tests
- Evaluating A/B Test Data using tests
	- t-test is a statistical test that calculates the size of the difference between two means given [their variance and sample size] noise in the data
		- t = (y1_mean - y2_mean)/(s1^2/N1 +s2^2/N2)^(1/2)
		- y1_mean, y2_mean are the central tendancies of the two datasets
		- s1, s2 are the standard deviation of the datasets
		- N1, N2 are the sizes (number of individuals) in th two datasets

		- larger t values indicate more significant differences in the means relative to the noise and lead to small p values--> the two samples in question were not drawn from the same population; 
		- depending on the problem, we choose a threshold of improbability called a significance level and if the p-value is smaller than alpha, there is a significant differences between the two sets of samples
### Assignment 1-4-7: Null-hypothesis significance and testing
- Null Hypothesis testing
	- tester has
		- a hypothesis that desribes what they think the data will look like if ther expectations are confirmed
		- a null hypothesis that describes what the data will look like if expectations are not confirmed
	- data is then compared to the null hypothesis
		- calculate a t-value  which is situated in a t-distribution that represents the tvalues you would get if the null hypothesis were true; the farther the calculated t-value is from the center, the less likely that the null hypothesis is true
			- the total area under the curve defined by our t-value sums to the p-value!
			- the p-value represents the probability of getting a t-value that is large or larger if the null hypothesis is true
	- p<.05
		- is the rule of thumb; that means that there is a  1 in 20 chance of returning a false positive
		- corresponds to the two sigma (standard deviation) mark
		- not ubiquitous; there are fields where you have to be much more sure than p = .05 and the threshold value will be much lower
		
### Assignment 1-4-8: T-tests and Philosophy of NHST
- T-values
	- the default t-test is two tailed, which is to say it's the probability of getting a more extreme value in either direction.  
	- if a negative result is impossible for some reason (for example) one can use a one-taled t-test
- Philosophy of NHST (Null Hypothesis significance testing)
	- p-value represents the probability of getting the data you have if the null hypothesis were true in the population; put another way, the probability of pulling this data by random chance from the same population that is unaffected
	- no mention of an "actual hypothesis", but rather an acceptance of "not not rejectign the null hypothesis" is tantamount to accepting the hypothesis
	- However, you can't truly limit the possibility space to two outcomes so we can't prove taht our hypothesis occured adn that the effect wasn't due to some other factor.  Instead we stick with disproving a null hypothesis and stating that the results support the hypothesis we put forward
	
### Assignment 1-4-9: Experiementation guided example