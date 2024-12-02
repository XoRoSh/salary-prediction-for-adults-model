from bnetbase import Variable, Factor, BN
from itertools import product
import csv

persons = []

def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    var_domains = [variable.domain() for variable in factor.get_scope()]
    products = list(product(*var_domains))
    new_factor = Factor(factor.name + ", normalized", factor.get_scope())

    sum = 0
    for product1 in products: 
        value = factor.get_value(list(product1))
        sum += value

    for product1 in products: 
        value = factor.get_value(list(product1))
        new_value = value/sum
        new_factor.add_values([list(product1) + [new_value]])
    return new_factor 


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    '''
    scope = factor.get_scope()
    index = scope.index(variable)
    new_scope = scope.copy()
    new_scope.pop(index)
    
    var_domains = [variable.domain() for variable in new_scope]  #[[1, 2, 3], ['a', 'b'], ['heavy', 'light']] 
    products = list(product(*var_domains))

    # var Weight, val = heavy
    all_values = []
    for product1 in products: #[1, 'a'], [2, 'a'], [3, 'a']
        variable_values = list(product1) #[1, 'a']
        variable_values.insert(index, value) #[1, 'a', 'heavy']
        factor_value = factor.get_value(variable_values) #P(1, 'a', 'heavy')
        variable_values.pop(index) #[1, 'a']
        variable_values.append(factor_value) #[1, 'a', P(1, 'a', 'heavy')]
        all_values.append(variable_values) #[[1, 'a', P(1, 'a', 'heavy')], [2, 'a', P(2, 'a', 'heavy')], [3, 'a', P(3, 'a', 'heavy')]]
                                           #[[1, 'b', P(1, 'b', 'heavy')], [2, 'b', P(2, 'b', 'heavy')], [3, 'b', P(3, 'b', 'heavy')]]
    
    new_factor = Factor(factor.name + "," + variable.name + "-restricted", new_scope)
    new_factor.add_values(all_values)
    return new_factor






def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    scope = factor.get_scope()
    index = scope.index(variable)
    new_scope = scope.copy()
    new_scope.pop(index)

    var_domain = variable.domain()

    var_domains = [variable.domain() for variable in new_scope] 
    products = list(product(*var_domains))

    new_factor = Factor(factor.name + ", sumout " + variable.name, new_scope)

    values = []
    for product1 in products:
        assignment = list(product1)
        sum = 0
        for i in var_domain: 
            assignment.insert(index, i)
            sum += factor.get_value(assignment) 
            assignment.pop(index)
        values.append(list(product1) + [sum])
    new_factor.add_values(values)

    new_factor.add_values(values)

    return new_factor 


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    #[[1, 2], ['a', 'b'], ['heavy', 'light']] 
    #[['heavy'] ['light'], ['cyka'] ['blyat']]
    # [1, 'a', 'heavy', "cyka"] [1, 'a', 'heavy', "blyat"]
    # [1, 'a', 'light', "cyka"] [1, 'a', 'light', "blyat"]
    # [1, 'b', 'heavy', "cyka"] [1, 'b', 'heavy', "blyat"]
    # [1, 'b', 'light', "cyka"] [1, 'b', 'light', "blyat"]
    # [2, 'a', 'heavy', "cyka"] [2, 'a', 'heavy', "blyat"]
    # [2, 'a', 'light', "cyka"] [2, 'a', 'light', "blyat"]
    # [2, 'b', 'heavy', "cyka"] [2, 'b', 'heavy', "blyat"]
    # [2, 'b', 'light', "cyka"] [2, 'b', 'light', "blyat"]
 
    


    raise NotImplementedError

def ve(bayes_net, var_query, EvidenceVars):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by EvidenceVars. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param EvidenceVars: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    For example, assume that
        var_query = A with Dom[A] = ['a', 'b', 'c'], 
        EvidenceVars = [B, C], and 
        we have called B.set_evidence(1) and C.set_evidence('c'), 
    then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26]. 
    These numbers would mean that 
        Pr(A='a'|B=1, C='c') = 0.5, 
        Pr(A='a'|B=1, C='c') = 0.24, and 
        Pr(A='a'|B=1, C='c') = 0.26.

    '''
    raise NotImplementedError


def naive_bayes_model(data_file, variable_domains = {"Work": ['Not Working', 'Government', 'Private', 'Self-emp'], "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'], "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'], "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'], "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], "Gender": ['Male', 'Female'], "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'], "Salary": ['<50K', '>=50K']}, class_var = Variable("Salary", ['<50K', '>=50K'])):
    '''
   NaiveBayesModel returns a BN that is a Naive Bayes model that 
   represents the joint distribution of value assignments to 
   variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
   assumes P(X1, X2,.... XN, Class) can be represented as 
   P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
   When you generated your Bayes bayes_net, assume that the values 
   in the SALARY column of the dataset are the CLASS that we want to predict.
   @return a BN that is a Naive Bayes model and which represents the Adult Dataset. 
    '''
    variable_domains = {
        "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
        "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
        "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
        "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
        "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
        "Gender": ['Male', 'Female'],
        "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
        "Salary": ['<50K', '>=50K']
    }

    variables = []
    for key, domain in variable_domains.items():
        variables.append(Variable(key, domain))

    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    # print("headers = ", headers)
    # print("input data", input_data)

    # Count occurrences for each header
    header_counts = {header: {} for header in headers}
    print(header_counts)



    person = {header: {} for header in headers}

    # Count occurences  
    total = 0
    for row in input_data:
        for i, value in enumerate(row):
            total+=1

            person[headers[i]] = value
            persons.append(person)

            if value in header_counts[headers[i]]:
                header_counts[headers[i]][value] += 1
            else:
                header_counts[headers[i]][value] = 1
    total = total/len(header_counts)
    # print(header_counts)



    # Create variables
    variables = []
    factors = []
    for variable in header_counts.keys(): 
        domain = header_counts[variable]
        var = Variable(variable[:3], list(domain.keys()))
        variables.append(var)
        print(var)
        factor = Factor(f"P({var.name})", [var])












    gender = Variable("Gender", ['Male', 'Female'])
    F1 = Factor("P(G)", [gender])
    F1.add_values(
    [['Male', 0.676],
    ['Female', 0.324]])


    





    # work = Variable("Work", ['Not Working', 'Government', 'Private', 'Self-emp'])
    # F2 = Factor
    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
#     raise NotImplementedError


def explore(bayes_net, question):
#     '''    Input: bayes_net---a BN object (a Bayes bayes_net)
#            question---an integer indicating the question in HW4 to be calculated. Options are:
#            1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
#            2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
#            3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
#            4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
#            5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
#            6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
#            @return a percentage (between 0 and 100)
#     ''' 


    

    if question == 1:
        #1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
        # 1. Woman, E1 = Married, E2 = Not-Married

        pass
    elif question == 2:
        pass
    elif question == 3:
        pass
    elif question == 4:
        pass
    elif question == 5:
        pass
    elif question == 6:
        pass
    else:
        raise ValueError("Invalid question number")

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))

