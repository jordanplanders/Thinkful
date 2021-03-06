# Unit 1
## Lesson 1: Intro to Python
### Assignment 1: What is programming?
- program: set of instructions for a computer to execute
	- programming (1) finds solutions to problems (2) implements solutions in a language that a computer can understand and execute
- Fibonacci @example
	- algorithm (pseudo code)
		- n = iterations
		- start with list with first two entries= 0, 1
		- iterate from 2 to n+2 using fib_num[n-2] and fib_num[n-1] to get the new fib_num
	- boils down into three parts: (1) getting the number of iterations from user, (2) computing the numbers, (3) printing the list
	- formalized code (apparently) includes one function for each process wrapped with a main() function and run as "main()" at the bottom of the script (oy)	
		- @question is this really what good practice is? It seems like a lot of overhead for a problem like this...
### Assignment 2: What is Python?
- Python is worth learning because the options are extensive and it reads like English
- high level language @readmore
### Assignment 3: Variables
- variable: name that is attached to a value
	- manage the state of a program (values of set of variables used)
- assigning a value: use the = operator
- accessing: refer to the value as needed
- naming conventions:
	- must begin with a letter or underscore but may be followed with letters, numbers or underscores (python is case sensitive)
	- cannot be a reserved word, e.g. string
	- generally use snake_case except for Classes (TitleCasing) and constants (ALLCAPS)
### Assignment 4: Data types
- Strings: textual data delineated by quotes
- numbers: integers and floats (floating point arithmetic is a tricky subject)
- booleans: True, False
- None: lack of value
- type conversion
	- can convert string numbers to floats or ints
	- can convert floats or ints to strings
### Assignment 5: Collections of Data
- lists: ordered collections of data; can contain anything and any combination of objects, can be accessed with bracket notation by index
- dictionaries: key-value pairs wrapped in curly braces
- sets: unordered lists
## Lesson 2: Functions, Strings, and Numbers
### Assignment 1: Function Basics
- function: repeatable process or behavior 
- argument: a value passed to a function
- defining and executing functions
	- (1) def signifies that a function is about to be defined (2) followed by () that contain parameter list (parameters expected by the function when it is called in the code) and (3) : , then (4) indented instructions to do something (5) probably a return statement 
		- note: python is sensitive to whitespace 
	- Calling functions: call the function by its name with expected parameters in parentheses 
	- in math a function maps each input to a single output, but in programming functions are just a set of repeatable instructions that don't have to return anything
		- a "determinate" function maps arguments to exactly one return value
		- a function that is pure, has no side effects)
		- a function that is determinant and pure is akin to a mathematical function 
### Assignment 2: Working with Strings
- sequence of characters enclosed by either single or double quotes (tripple quoted denote an extended comment)
- special characters: to use special characters (e.g. &, %, #, ', " etc) precede with a backslash (called escaping). \n is a new line and \t is a tab
- concatenating and repeating: can use math operators + for concatenating and * for repeating 
- indexes and sliceing: string[ik] refers to the letter of the string at index ik, string[ik:ij] revers to the letters from index ik up to but not including index ij (string[:ik] and string[ik:] will return all the characters in the string up until index ik and all the letters including and after index ik, respectively). Negative numbers start counting from the end of the string, with the last character being -1
- comparing strings: use "==" to compare two strings
- string methods: string.method(); methods are called by following the string with a . and then the method name and then parentheses with any arguments 
	- string_with_braces_{ }.format(value)
### Assignment 3: String Drills (Codewars)
### Assignment 4: Working with Numbers
- Python 3 can represent numbers of any size, floats are double precision binary floating point numbers that represent decimals as preceisely as possible in 52 bits. Tread carefully when comparing decimals because .2+.4 ==.6 will throw back _False_ 
- operations: 
	- addition
	- subtraction
	- multiplication
	- division
	- floor division
	- modular arithmetic (modulo operator)
	- exponentiation
- note: in programming the = operator assigns a value to a variable rather than being a point of comparison (thus x=x+1 is not nonsense)
- compound operators
	- += add and assign
	- -= subtract and assign
	- *= multiply and assign
	- /= divide and assign
	- //= floor divide and assign
	- %= modulo and assign
	- **= exponentiate and assign
- comparison operators
	- < less than
	- > greater than
	- <= less than or equal to
	- >= greater than or equal to
	- == equal to
	- != not equal to
- math library (examples)
	- math.floor()
	- math.sqrt()
	- math.py
	- math.e
### Assignment 5: Number Drills (Codewars)
## Lesson 3: Application Logic
### Assignment 1: Making Logical Assertions
- True: anything that is a thing (e.g. True, string, number, list, dict)
- False: anything that is nothing (e.g. False, None, 0, empty string, empty list, empty dict)
- Logical operators: 
	- and: both must be true
	- or: one or more must be true
### Assignment 2: Control Flow and Conditionals
- Conditional statements: if, elif, else
- Exception handling: try and except to try to run something and then offer a way to deal with an error if one is thrown and gracefully continue running the code
### Assignment 3: Logic Drills (Codewars)
## Lesson 4: Lists and Loops
### Assignment 1: Lists
- lists can strore a collection of data in an ordered sequence, data of any type and can be of mixed type
#### Section 1: Creating lists
- can initiate a list either as an empty list [] or with items in it seperated by commas [1,2,3,4]
- calling list() on an iterable will make the iterable into a list of the items
#### Section 2: Accessing 
- access list items by index (e.g. lst[0] is the first item in the list lst)
- accessing list by slicing (e.g. lst[:3])
#### Section 3: Updating a list
- replacing the item (e.g. lst[0] = 'bear')
- add an item to the end with .append()
- pop an item off the front of a list (returns the item popped)
- insert an item into a list with .insert(index, value)
#### Section 4: Additional list methods
- find the index of a particular value with .index(value)
- sorting a list with .sort will sort the list alphabetically or in number order
- len() will return the length of the list
### Assignment 2: Loops and iteration
- allow you to execute a set of instructions a specific number of times or until some condition is met
#### Section 1: for loops
- run through a block of code a specific number of times over an iterable
- for ik, value in enumerate(lst):
- for key in dict.keys():
- for ik in range(len(list)):
#### Section 2: while loops
- useful when you don't know how many times you need to run the loop but know that there is a logical condition that will mark an appropriate stopping place
- while [some statment that is initially false but will become true]:
	- often add an escape hatch with if [alternative condition]: break
### Assignment 3: Loops and Lists Drills (Codewars)
## Lesson 5: Dictionaries
### Assignment 1: Working with Dictionaries
- collections of items in curly braces {}, where an item is a key: value pair
- keys can be strings or immutable objects (e.g. sets)
- values are accessed by d[key]
- to add a value: d[key]= value, to update a value just reassign: d[key]= value2
- to delete an item: del d[key]
- boolean operations
	- assume d = {key1: value1, key2: valeu2}
	- key1 in d --> True
	- key3 not in d --> True
	- key3 in d --> False
- methods and looping
	- dictionary view object of keys: d.keys()
	- dictionary view object of values: d.values()
	- dictionary view object of key-value sets: d.items()
	- notes about view objects: 
		- can make view objects into lists, but as is, they are dynamic and will reflect the up-to-date version of the dictionary
		- dictionaries and their view objects are unordered, lists are ordered
### Assignment 2: Dictionary Drills (Codewars)
## Lesson 6: Objects, classes, and modules
- Data Science isn't a very object oriented discipline, but good to know how objects work
### Assignment 1: Intro to Objects
- object: collection of attributes, each with a name and a value, an id function and a type function; can use dir() to get the attributes of an object
- methods are funcions taht are attached to an object as attributes
- attributes can also be values
### Assignment 2: Classes, types, and inheritance
- abstraction and classes:
	- class: all the attributes that instances share with one another
	- abstraction: there are classes like "number" and "rational" that integers and floats belone to.  Integers and floats are said to inherit attributes from these classes, in addition to the attributes that are unique
	- format: 
			class CamelCaseName(subclass):
				def _ _ init _ _ (self, value_in1, value_in2)
					self.value_in1 = value_in1
					self.value_in2 = value_in2
				value = num

				def _ _ repr _ _(self):
					return 
		- notes: python automatically calls init method when an instance is created, repr is a method that tells your object how to interact with print()
### Assignment 3: Modules
- modules or packages are additions to the python code base that  you can import on an "as needed" basis
	- import [package]- import statments go at the top of the code and can also be "import [package] as [alias]" 
	- it is also possible to import a particular class or function from a module, e.g. "import [package.class or function] as [alias]"
	- to make module executable from the command line via "python [module name].py", add: if _ _ name _ _ == "_ _ main _ _":

	[last exercise that I couldn't submit for some reason]
	
		class Quark(object):
		    def __init__(self, color, flavor):
		        self.color = color
		        self.flavor = flavor
		    
		    baryon_number = 1/3
		    
		    def interact(self, quark):
		        temp_color = self.color
		        self.color = quark.color
		        quark.color = temp_color
		        
		        temp_flavor = self.flavor
		        self.flavor = quark.flavor
		        quark.flavor = temp_color
### Assignment 4: Object Drills (Codewars)
        
# Unit 2
## Lesson 1: Introduction to NumPy and Pandas
### Assignment 1: Numpy
#### Section 1: Arrays
- primary data structure of numpy is an array
- create an array by calling np.array([])
- create an array of arrays by calling np.arange([]).reshape(rows, columns) or np.array([],[])
#### Section 2: Element-wise and Aggregator functions
- element-wise functions: e.g. np.square(), np.sqrt(), np.cos() will all apply function to the elements of the array
- aggregator functions: e.g. np.min(), np.max(), np.mean(), np.std() will apply function to array and return a single summary statistic
### Assignment 2: Pandas Data Frames
#### Section 1: The Data Frame 
- dataframes are similar to np.array() but have column and row labels
- create a dataframe 
	- calling pd.DataFrame() on a np.array([],[])
	- label columns by df.columns = [] and rows by df.index = []
	- can also do the entire thing in one go: pd.DataFrame(np.array([],[]), columns=[], index=[])
- adding more data
	- in generality: df['column_name']=[list_of_values]
	- df = pd.DataFrame(index=['rowlabel1', 'rowlabel2', 'rowlabel3'])
	- df['col1'] = [1,2,3]
	- df['col2'] = [4,5,6]
- handling data
	- can address columns with either dot notation df.col1 or df['col1']
	- can do an element-wise calculation to see output, or can assign to a new df column
### Assignment 3: Pandas-Selecting and Grouping
- most basic form of selecting is df['variable_name'] which returns a series of data, and to retrieve row labels, need to use .index
#### Section 1: Basic Selects with .loc and .iloc
- df.loc[row_name] will return the row
- df.iloc[:, 'variable_name'] will return the column
- to select the value of the variable associated with 'variable_name' and row_name, use df.loc[row_name, variable_name]
- to do integer indexing can do df.iloc[row_1:row_n, col_n]
#### Section 2: Conditional Selection
- use lambda functions e.g. df2.loc[lambda df: df2[variable_name]>3, :] which says return a dataframe where where the valuse of df2[variable_name] >3
- alternatiely: df2[df2[variable_name]>3] would also work
	- multiple conditions: df2[(df2[variable_name]>3 & df2[variable_name2]<5)]
#### Section 3: Groups
- create small dataframes that are made up of grouped rows with df2.groupby('variable_name')
- can be useful to have it return an aggregate measure, something like df2.groupby('variable_name').aggregate(np.mean)
### Assignment 4: Working with Files
#### Section 1: Opening files with and loading into Pandas
- CSV
	- load: pd.read_csv(); options for managing headers etc.
	- output: pd.to_csv('my_data.csv'); write dataframe to csv
- JSON and XML are semistructured data forms 
	- JSON (Javascript Object Notation): key value pair that can be very nested and are delivered as strings; open with pd.read_json('') @question does it balk if the json is nested?
	- XML (extensible markup language) similar to html in that is managed by opening and closing tags; no pandas option so have to write a parser to convert them to en element tree  and process it in to lists that can be fed into Pandas. (also consider lxml package)
		- import xml.etree.ElementTree as ET
		- tree = ET.parse('purchases.xml')
- Python open()
	- with open(' ') as f: text = f.readlines()
		- using open() with "with" automatically closes the file once the loop is over
- Encoding
	- Python3 strings are unicode and use utf-8 encoding 
	- however... sometimes you have to fish around and try different encoding schemes to extract text from the binary (alert: if the file came from Windows, try cp1252, and if that fails try Chardet to suss out what the encoding scheme is)
#### Section 2: Tuning Dataframe
- set index column: df = df.set_index('column_name')
- change column name: df = df.rename(columns = {old_name : new_name})
#### ipython notebook: https://github.com/jordanplanders/Thinkful/blob/master/Unit%202/2-4-1.ipynb
## Lesson 2: Data Visualizations with matplotlib
### Assignment 1: Basic Plots and Scatter
#### Section 1: Matplotlib
- use matplotlib for most plotting, and we are partiularlay interested in pyplot
- import matplotlib.pyplot as plt
- %matplotlib inline (to show plots)
- plot types:
	- plt.plot[1,2,3,4] will plot lines between these points with the x axis being the index of the poing and the numbers being the y value
		- color = 
	- plt.scatter(x= [], y = []) will produce a plot of x, pairs, not connected by lines
		- color = 'purple' %color
		- marker = 'x' %marker style
		- s = 20 %marker size
	- annotations: 
		- plt.ylim([]) %sets y limits
		- plt.xlim([]) %sets x limits
		- plt.ylabel('') %sets y axis label
		- plt.xlabel('') % sets x axis label
		- plt.title('') %sets plot title
#### Section 2: Pandas Plot
- plot directly using pandas
- df.plot(kind = (e.g. 'scatter', 'line'), x = 'variable name', y = 'variable name')
### Assignment 2: Subplots
- subplots are a way fo amkeing multiple plots in one figure
- Create a figure obect:
	- plt.figure(figsize= (10, 5))
- Add subplots
	- plt.subplot(1,2,1)
	- plt.plot(......)
	- plt.subplot(1,2, 2)
	- plt.plot(......)
	- plt.tight_layout()
- Other ways
	- fig, ax = plt.subplots(n,figsize=(6,6)) %n= number of subplots
	- ax = fig.add_subplot(111, aspect='equal') %111 is position
### Assignment 3: Statistical Plots
#### Histograms: 
- plt.hist(x); histograms take range nd divide it into evenly sized bins and count how many points fall in each bin and reflect a count (in terms of the height of the column of the number of points that fell in each bin)
	- bins = n %can specify the number of bins --specifies how many bin, not where bin edges are
	- bins = np.arange(-10, 40 ) % make bins that are at integer makrs from -10 to 40
	- normed = True % scales the bins of a given distribution so that the area under each curve sums to 1, allowing one to campare the central tedancy and the skew etc, more effectively
	- color = color % can specify the bin color
	- alpha = flaot % opacify
#### Boxplot:
- takes an array and summarizes it into average value, standard deviation, quartiles, with a few outliers explicitly plotted
	- box is the 25th percentile to the 75th percentil with a line for the median value
	- whiskers extend to +/- 1.5x interquartile range
	- any outliers (fliers) are plotted seperately
- useful to comparing statistics/spread/ranges, but not density  
### Assignment 4: Challenge- What do you see? @exercise
Let's go out into the world and generate some beautiful visuals. Pick a data source from this aggregation, load the data into a pandas data frame, and generate a series of visuals around that data using pyplot.

Each visualization should be accompanied by 2-3 sentences describing what you think is revealed by this representation. Generate at least four different visuals, and be sure to use different types as well as the subplot functionality discussed above. And remember: clean and elegant visuals are key to telling a coherent story.

Collect your images and descriptions into a shareable format. A Jupyter notebook is best, but anything you can link to is fine, including Google docs or markdown files on GitHub or gists, and share the link below.
### Assignment 1: Basic Plots and Scatter
#### Section 1: Matplotlib
- use matplotlib for most plotting, and we are partiularlay interested in pyplot
- import matplotlib.pyplot as plt
- %matplotlib inline (to show plots)
- plot types:
	- plt.plot[1,2,3,4] will plot lines between these points with the x axis being the index of the poing and the numbers being the y value
		- color = 
	- plt.scatter(x= [], y = []) will produce a plot of x, pairs, not connected by lines
		- color = 'purple' %color
		- marker = 'x' %marker style
		- s = 20 %marker size
	- annotations: 
		- plt.ylim([]) %sets y limits
		- plt.xlim([]) %sets x limits
		- plt.ylabel('') %sets y axis label
		- plt.xlabel('') % sets x axis label
		- plt.title('') %sets plot title
#### Section 2: Pandas Plot
- plot directly using pandas
- df.plot(kind = (e.g. 'scatter', 'line'), x = 'variable name', y = 'variable name')
### Assignment 2: Subplots
- subplots are a way fo amkeing multiple plots in one figure
- Create a figure obect:
	- plt.figure(figsize= (10, 5))
- Add subplots
	- plt.subplot(1,2,1)
	- plt.plot(......)
	- plt.subplot(1,2, 2)
	- plt.plot(......)
	- plt.tight_layout()
- Other ways
	- fig, ax = plt.subplots(n,figsize=(6,6)) %n= number of subplots
	- ax = fig.add_subplot(111, aspect='equal') %111 is position
### Assignment 3: Statistical Plots
#### Histograms: 
- plt.hist(x); histograms take range nd divide it into evenly sized bins and count how many points fall in each bin and reflect a count (in terms of the height of the column of the number of points that fell in each bin)
	- bins = n %can specify the number of bins --specifies how many bin, not where bin edges are
	- bins = np.arange(-10, 40 ) % make bins that are at integer makrs from -10 to 40
	- normed = True % scales the bins of a given distribution so that the area under each curve sums to 1, allowing one to campare the central tedancy and the skew etc, more effectively
	- color = color % can specify the bin color
	- alpha = flaot % opacify
#### Boxplot:
- takes an array and summarizes it into average value, standard deviation, quartiles, with a few outliers explicitly plotted
	- box is the 25th percentile to the 75th percentil with a line for the median value
	- whiskers extend to +/- 1.5x interquartile range
	- any outliers (fliers) are plotted seperately
- useful to comparing statistics/spread/ranges, but not density  
### Assignment 4: Challenge- What do you see? @exercise
Let's go out into the world and generate some beautiful visuals. Pick a data source from this aggregation, load the data into a pandas data frame, and generate a series of visuals around that data using pyplot.

Each visualization should be accompanied by 2-3 sentences describing what you think is revealed by this representation. Generate at least four different visuals, and be sure to use different types as well as the subplot functionality discussed above. And remember: clean and elegant visuals are key to telling a coherent story.

Collect your images and descriptions into a shareable format. A Jupyter notebook is best, but anything you can link to is fine, including Google docs or markdown files on GitHub or gists, and share the link below.

https://github.com/jordanplanders/Thinkful/blob/master/Unit%202/2-2-4.ipynb

# Unit 3 Statistics for Data Science
## Lesson 1: Summarizing data
### Assignment 1: Population v. Sample
- population is the full set of entities with some characteristic 
- sample is a randomly selected subset of the population 
### Assignment 2: Measures of Central Tendency
- statistics can describe either an individual variable or the relationship among two or more variables 
- variable: information about a particular measurable concept; each measurement is a datapoint
#### Section 1: Central Tendency
point around which datapoints in a variable cluster
- mean (expected value) @question it's better to use numpy than regular python, yes?
	- NB: sensitive to extreme values
- median (middle value when the values are ordered from least to greatest (or average of two middle numbers))
	- import statistics; statistics.median(list)
	- np.median(array)
	- NB: isn't sensitive to extreme values
- mode (variable that occurs most frequently)
	- import statistics; statistics.mode(array) [will throw an error if there exists multiple modes]
	- (values, counts) = np.unique([], return_counts=True); ind = np.argmax(counts); values[ind]
- bias: an estimate is considered unbiased if, across multiple representative samples, the sample estimates converge on the population value @question what statistic would lead to a biased estimate?
### Assignment 3: Measures of Variance
#### Section 1: Variance
- how much values differ from the central tendency; if they differ very little, he variance is said to be low
- how valuable is each datapoint within a variable; low variance means that each datapoint adds relatively little new information about the concept being measured
- in data science we care a great deal about variance because one of the primary topics of interest is whether to quantities are different; a variable with lots of variance provides information about the differences between observations that can be used to understand and predict future outcomes. 
- sum((x-mean)**2)/(n-1)
	- penalize more for extreme values with the square, plus squaring the distance gets rid of the sign problem
	- divide by n-1 because dividing by n would underestimate population variance @question there was a proof about why we use n-1 rather than n....
	- np.var(array), df[''].var() will also calculate variance
#### Section 2: Standard Deviation
square root of the variance; quantifies the variance in the population
- np.std([], ddof=1) "delta degrees of freedom" has to be included to account for sample standard deviation rather than population standard deviation (e.g. n-1 rather than n)
#### Section 3: Standard Error
quantifies uncertainty in the estimate of the sample mean
- if you were to run the same experiment again on this population with a different sample population one might expect a difference in the mean by as much as x%
- se = s / (n**.5), np.std(array, ddof=1)/np.sqrt(len(array)) @question why do we use n here rather than n-1?
#### Comparing 
- good exercise to create two populations, one with high variance, one with low variance, plot them as histograms, then sample them randomly for n=100 and calculate mean, standard deviation, standard error
### Assignment 4: Describing Data with Pandas
most standard summary statistics have already been coded up in numpy or pandas so no need or reinvent the wheel
#### Section 1: Methods
- .describe(); df.variable.describe() will spit out a dataframe of summary statistics of number fields
	- sometimes better to us df.groupby(variable).describe() instead so that the summary statistics are reflect the groups they are associated with rather than spanning the whole dataset
- .mean()
- .std()
- .value_counts(); df.variable.value_counts() returns the counts of values of a categorical variable. Also works on non categorical variables, but not as useful for continuous variables except to find null values
### Assignment 5: Drill-Describing Data 
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-2-5.ipynb

## Lesson 2: Basics of Probability
quantifying the likelihood of a future outcome
- frequentist: describing how often a particular outcome would occur in an experiment, if that experiment were repeated over and over
- Bayesian: describing how likely an observer expects a particular outcome to be in the future based on previous experience and expert knowledge
	- prior probability, or prior = prior experience
	- posterior probability = updated probability based on newest experiment
	most of the time both approaches converge on the same answer
	Comparison:
	- frequentists are trying to calculate the likelihood of getting the data you have in the context of a fixed of an existent (if unknown) population value
	- Bayesians are trying to calculate the most likely population value given the data you have and update that value as data changes
### Assignment 2: Randomness, Sampling and Selection Bias
- randomness: state in which all possible outcomes are equally likely; using randomness each element of a population has an equal chance of being chosen which allows a sampler to pull a sample that (hopefully) resembles the population at large
	- the more variable the population, the larger the sample has to be to be representative (though it can still randomly be biased)
	- Hiccough: random sampling depends on perfect access to the population (rare); 
		- selection bias: systematic differences between the sample and the population
			- example: Landon v. Roosevelt election and the Literary Digest polling error (randomly selected using phone numbers, ignoring the fact that phones were a luxury item during the depression)
### Assignment 3: Independence and Dependence
an event is independent of other events in the sample space if the outcome of that event is not affected by the outcome of other events in the sample space.  
- ex: suppose there is a bag of marbles with 5 blue and 5 red.  On the first draw, the probability of drawing a red (P(red)) is 1/2.  If you replace the marble, the second draw will be unaffected by the first draw and P(red)=P(blue)=1/2
- The probability two or more independent events is the multiplication of their probabilities (e.g. picking a red and then picking a blue assuming replacement: P(red intersection blue)= P(red) * P(blue)= 1/2 * 1/2 = 1/4)
When the probability of event B changes based on the outcome of event A, event B is said to be dependent or conditional on event A. (e.g. pulling a marble without replacement)
- ex: the probability of pulling the first marble will be 1/2, but not the second.  The second is P(red|blue) which is read the probability of drawing a red marble conditional on a blue event = 5/9 (there are still 5 red marbles, but only 9 marbles left).  Also, the probability of drawing a blue marble is 4/9.  (Note that the se probabilities always add to 1)
- two conditional events like the probability of a red event given the last event was a blue event: P(blue) * P(blue|red) = 1/2 * 5/9. Alternately, the probability of two blue events in a row would be P(blue) * P(blue|blue) = 1/2 * 4/9
	- formalized: P(A intersection B) = P(A) * P(B|A) @question How does this extend to n events?
### Assignment 4: Drill- Exercises in Probability
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-2-4.pdf
### Assignment 4: Bayes' Rule
- Conditional Probability: we aren't focused on the probability of an independent event
	- e.g. probability that you're infected with a disease given a positive test
- The rule itself:
	- P(A | B) = P(B | A) * P(A) / P(B)
	- P(A|B) = P(B|A) * P(A)/ [P(A) * P(B|A) + P(~A) * P(B|~A)]
	in English:
		the probability of A given B **: P(A | B)** = the probability of B given A: **P(B | A)**, times the probability of A: **P(A)**, divided by the probability of B: **P(B)**. We expand the probability of B **P(B)** in the second formula where A~ is the inverse of A, so in our case not being infected. The numerator is our scenario of interest, while the denominator represents the realm of scenarios that could give our condition.
	put another way:
	- P(Infected| Positive Test) = P(Positive Test| Infected) * P(Infected) / P(Positive Test)
	- @question There is a "percent times amount" thing going on here, but I can't quite put my finger on it. there's a denominator that is percent subjects in A * probability of B given A + percent subjects not in A* probability of B given not in A...
	and a third...
	To apply Baye's rule we need to know the following: 
	- A label for the state/situation/event we want to calculate the probability of.  Call this *H*
			ex: MHP: there is a car behind door 1, the door you chose
	- A label for the observations/evidence that would inform the probability, *E*
			ex: MHP: Monty Hall shows you a goat
	- The probability that the state/situation/event *H* is true (this is the prior which will be updated in light of other information)
			ex: MHP: Each door has an equal starting probability; *P(H)=1/3*
	- The probability of the observations/evidence, *E*, given the claim, *H*, is true: *P(E|H)*
			ex: MHP: The probability that Monty Hall shows you a goat given there is a car behind door 1; (*P(E|H)=1*)
	- The probability of the observations/evidence (*E*) given the claim (*H*) is false: *P(E|~H)*
			ex: MHP: The probability that Monty Hall shows you a goat given there is a goat behind door 1; (*P(E|~H)=1*)

from random import shuffle
from random import choice

def new_exp():
    car_door = choice(doors)
    pick_door = choice(doors)
    return car_door, pick_door
    
def open_first_door(pick_door, car_door, doors):
    for door in doors:
        if door != pick_door and door != car_door:
            doors.remove(door)
            return doors

win_stay = 0
win_change = 0

total_stay = 0
total_change = 0
for ik in range(1000):
    doors = [0,1,2]
    car_door, pick_door = new_exp()
    doors = open_first_door(pick_door, car_door, doors)
    stay_go = choice(['go','stay'])

    if stay_go == 'stay':
        total_stay +=1
        if car_door == pick_door:
            win_stay += 1
    if stay_go == 'go':
        total_change +=1
        if car_door != pick_door:
            win_change += 1

print('changed then won:', win_change/total_change)
print('stayed then won:', win_stay/total_stay)

output:
changed then won: 0.6774847870182555
stayed then won: 0.31952662721893493
### Assignment 5: Drill- Exercises in Bayes' Rule
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-2-6.pdf
### Assignment 7: Evaluating Data Sources
- Bias: we can rarely evaluate all subjects of a population so we sample, but samples may or may not be truly random/representative.  Pay close attention to method of data collection
- Quality: unexplained blanks, uneven distribution over a dependent variable
- Exceptional Circumstance: was the data collected under an exceptional circumstance
- what to do?  Adjust (impute) or limit your conclusions
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-2-7.pdf
### Assignment 8: Challenge- Beware of Monty Hall
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-2-8.pdf
## Lesson 3: The Normal Distribution and the Central Limit Theorem
### Assignment 1: Define Normality
- most values clustered in the middle with symmetrical tails to the right and left such that 68% of values fall within one standard deviation of the mean, 95% fall within two standard deviations and 99.7% fall within three standard deviations. 
- PDF (probability density function) for a normal distribution is: 
		
		1/(2 * sigma^2 * pi)^(1/2)* e^(-(x-mu)^2/(2*sigma^2))
- easily summarized with two statistics (mean and standard deviation)
- common in nature
- requirement for many common scores (percentiles, z-scores) and statistical tests (t-tests, ANOVAs, bell-curve grading)
### Assignment 2: Deviations from Normality and Descriptive Statistics
- Real data isn't quite as normal as all that 
- tests of non-normality are sensitive to sample size so instead try:
	- QQ (quanitile-quantile plot): plot of a variable with an unknown distribution v. a variable with a known distribution. Both variables are sorted in ascending order and plotted with known distribution on the x-axis, and unknown on the y-axis. If they share a distribution then they should plot as a straight line from the lower left to upper right (can generate a distribution of same number of points for the known set)
	- histogram: can also plot histograms and will find that the distributions look different.  The mean may still fall where the data clusters but the standard deviation may not be meaningful (skewed distribution), and the average may not even be meaningful (as in a bimodal distribution)
### Assignment 3: Other distributions
#### Section 1: Bernoulli
- represents two possible outcomes of an event (e.g. a coin flip)
- f(k|p) = {p, if k=1; 1-p if k=0}
- NB: when a distribution is discrete (only has integer outcomes) it has a probability MASS function in contrast to a probability density function when its on a continuous scale
#### Section 2: Binomial
- counts the number of successe when an event with two possible outcomes is repeated many times; p, the probability of getting k successes during n repetitions
- f(k|n, p) = (n k)p^k(1-p)^(n-k) @question what is the math here again?
#### Section 3: Gamma
- time until an event when the event starts out unlikely, becomes more likely than becomes unlikely again
- summarized by a shape parameter (alpha) and an inverse scale parameter (beta)
- f(x|alpha, beta) = (beta^alpha * x^(alpha-1) * e^(-x * beta))/Gamma(alpha) for x>= 0 and alpha, beta >= 0
#### Section 4: Poisson 
- number of times a given event will occur during a given time interval
- lambda is the rate that events occur durng a given period
- yields a probability mass function
- f(k|lambda) = (lambda^k * e^(-lambda))/k!
#### Section 5: Conditional Distribution
- not hard to plot using pandas; simply plot histogram filtered against a condition
- plt.hist(df [ df[ 'variable_count'] ># ] [variable])
### Assignment 4: Drill- Descriptive Statistics and Normality
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-3-4.ipynb
### Assignment 5: CLT and Sampling
- while not all distributions are normal, as sample size increases, population parameters (e.g. mean) are still interpretable 
- Central Limit Theorem says (ish) that it is justifiable to start doing statistics on data sets of unknown distribution
### Assignment 6: Central Limit Theorem in Action
#### Section 1: Comparing Groups in a sample  (CLT in action)
- In order for us to compare to groups we either have to know that the two populations are normal or we have to be able to say that they are normal enough.  
- In comparing to samples, we need the mean and standard deviation of each (which both assume a level of normality).  We look at the difference between the means in the context of the combined variance; the larger the difference is relative to the variance, the less likely it is that the the difference is due to random chance and that the two samples were pulled from the same population and more likely it is that the samples were pulled from distinct populations. 
	- y_mean = x2_mean-x1_mean
	- standard_error or y = (sigma_1^2/n_1 + sigma_2^2/n_2)^(1/2) 
		- variance in the sameple differences around the population difference
	- t-value = y_mean/standard_error_y
		- in noisy sample the difference (y_mean) is likely to result from noise rather than real differnces in population mean
	- p-value = probability that a t-value this extreme would happen by chance; how likely it is that we would get the sampling data we observe if the two populations were not different form one another. p-value run from 0 to 1, with lower p-vaue indicating more confidence that there is a meaningful difference between the means of the two groups (that they come from two populations).
		- low p value --> different populations
		- high p value --> same population

		- likelihood of getting a difference this large or larger if the samples were from the same population 
### Assignment 7: Drill- Exploring the Central Limit Theorem
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-3-7.ipynb
## Lesson 4: Narrative Analytics
### Lesson 3: Narrative Analytics Guided Example
https://github.com/jordanplanders/Thinkful/blob/master/Unit%203/3-4-3.ipynb
- @question how does one go about arriving at the 8 or so mediums highlighted in the last figure?
- @question I don't think it makes sense to use a line plot for a plot of counts v. years
# Unit 4: Job
## Lesson 1: 
### Assignment 1: Explore the landscape 
- survey output: 
	- By creating quantitative descriptions of your data, you create insight that is a key deliverable for your team.
	- You interpret the meaningful reasons for features in a dataset.
	- You also pay attention to the detail of underlying assumptions, limits and exceptions when describing a system.
	- You are familiar with a variety of mathematical methods for describing dynamic systems and are highly skilled in using software that implements these.
	- You use a variety of graphical and numeric techniques to verify that you are delivering a high quality result that can be used to predict and optimise future performance.
	- When you are on the team, if there is information that can be gleaned from a system, you will find it.
	- Visualizer: 5.4, 
	- Communicator: 3
	- Data Wrangler: 4
	- Modeller: 6
	- Programmer: 4.5
	- Techonologist: 2.8
### Assignment 2: Survey the job market
- Job Postings: Survey the job postings in the city where you plan to work at the end of this bootcamp and pull out five job advertisements you're most excited about. Add them to your career planning document, along with a sentence about why you chose that particular job. Here are some suggestions of places to find data science job postings.  @question how do I do this if I don't feel ready for these jobs as I hope I will be in six months?
	- UCLA Programmer Analyst: https://www.glassdoor.com/Job/los-angeles-ca-programmer-analyst-jobs-SRCH_IL.0,14_IC1146821_KO15,33.htm?src=GD_JOB_AD&t=SEARCH_RESULTS&ao=4120&srs=MY_JOBS&jl=2908784094
	- UCLA Programmer Analyst: https://www.glassdoor.com/Job/los-angeles-ca-programmer-analyst-jobs-SRCH_IL.0,14_IC1146821_KO15,33.htm?src=GD_JOB_AD&t=SEARCH_RESULTS&ao=4120&srs=MY_JOBS&jl=2908784094
	- UC BUSINESS INTELLIGENCE DEVELOPER (PROGRAMMER/ANALYST IV): https://www.glassdoor.com/Job/los-angeles-programmer-analyst-jobs-SRCH_IL.0,11_IC1146821_KO12,30_IP2.htm
	- Bird Sr. Data Scientist: https://www.indeed.com/viewjob?jk=602e71b408bb1e82&tk=1cmha4nu2bqorb4k&from=serp&vjs=3
	- https://www.kaiserpermanentejobs.org/job/-/-/641/8371646
	- Factual Associate Data Scientist: https://www.indeed.com/viewjob?jk=80dfb247480ad09d&q=Data+Scientist&l=90019&tk=1cmh9vpddbqordve&from=web&vjs=3
		- About you:
			- You are comfortable with data analysis, wrangling, and curation, 
			- You have a degree in a quantitative field with coursework in statistics (e.g. Math, Linguistics, Physics, Chemistry, Engineering), 
			- You feel at home on the command line and with text processing
			- You are eager to learn new technologies and skills in data processing and analysis
		- Baseline Skills (these are required):
			- 1+ years of full-time work experience or domain-relevant internships
			- Proficient with Unix commands and Ruby/Python scripting
			- Familiar with regular expressions, DOM/CSS/HTML, parsing JSON, and information extraction
			- Experience in reporting, analytics and databases
		- Specialized Skills (you need expertise in at least one of the following): 
			- Experience with implementing machine learning pipelines
			- Experience with Spark or MapReduce
			- Experience with SQL and/or querying MongoDB or Solr Indexes
- Company-first research: Next, turn your attention away from job postings and towards companies, whether or not they're hiring. Your goal is to put together a short hit list of your top five dream companies. Add them, along with a sentence about what they do and why you chose each one, to your planning document. If any of these companies overlap with the job postings you found above, add additional companies to your hit list until you have five total novel companies listed here.
	- Lamont-Doherty Earth Observatory
	- Motional.ai: http://www.motional.ai/#tab2
	- GoodRx: https://angel.co/goodrx/jobs
### Write your own story
Think about your ideal career two or three years from now. What particular skills have you built? What techniques and tools are your specialty? What accomplishments might you have made? Is your title "Data Scientist", or something different? Are you working in a particular industry, or solving a particular class of problems?

For this final assignment you have writing prompts about the ideal job you envision above. Write responses to each and add them to your planning document.

- Describe your ideal job in one or two paragraphs as though you're talking with an industry professional. They're familiar with the business and with the culture, they know the jargon and the buzzwords. Focus on the skills, tools, and / or industry you hope to deeply specialize in.
- Describe your ideal job in one or two paragraphs as though you're talking with a non-technical family member or friend. Avoid jargon and keep things simple while remaining specific. If you went deep in your specialization above, emphasize here the breadth of skills and work.
- Synthesize the two exercises above into a short aspirational "professional summary" that might be appropriate on your LinkedIn profile, resume, or portfolio in the future.