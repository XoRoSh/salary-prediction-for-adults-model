from bnetbase import Variable, Factor, BN
from itertools import product
import csv

persons = []
multiply_verbose = False 
ve_verbose = False 

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

def multiply_2_factors(factor1, factor2):
    if multiply_verbose:
        print("")
        print("Scopes and Variables")
        print("factor1", factor1.get_scope())
        for i in factor1.get_scope():
            print("     ", i.name, i.domain())
        print("factor2", factor2.get_scope())
        for i in factor2.get_scope():
            print("     ", i.name,i.domain())

    new_scope = factor1.get_scope()
    for var in factor2.get_scope():
        if var not in new_scope:
            new_scope.append(var)

    new_scope.sort(key=lambda var: var.name)
    if multiply_verbose:
        print("Multiplied Factor",  new_scope)
        for i in new_scope:
            print("     ", i.name, i.domain())
        print("new scope", new_scope)

    
    product1 = list(product(*[var.domain() for var in factor1.get_scope()]))
    product2 = list(product(*[var.domain() for var in factor2.get_scope()]))
    if multiply_verbose:
        print("\n" * 1)
        print("Assignments: ")
    all_values = []
    for assignment1 in product1:
        value1 = factor1.get_value(list(assignment1))
        for assignment2 in product2:
            # Multiply 2 assignments 
            value2 = factor2.get_value(list(assignment2))
            value_res = value1 * value2
            if multiply_verbose:
                print(assignment1, assignment2, value_res)

            # Sort assignment
            new_assignment = list(assignment1) + list(assignment2)
            new_assignment.sort(key=lambda x: str(x).replace('-', ''))

            valid_assignment = True
            to_remove = []
            for i in range(len(new_assignment) - 1):
                if new_assignment[i] in new_scope[i].domain(): 
                    if new_assignment[i+1] in new_scope[i].domain():
                        if multiply_verbose:
                            print(new_assignment[i], new_assignment[i+1])
                        first = new_assignment[i]
                        second = new_assignment[i+1]
                        to_remove.append(new_assignment[i])
                        if first != second:
                            if multiply_verbose:
                                print("INVALID ASSIGNMENT")
                            valid_assignment = False
                            break
                        else: 
                            if multiply_verbose:
                                print("VALID ASSIGNMENT")
        
            if valid_assignment:
                for item in to_remove:
                    new_assignment.remove(item)
                all_values.append(new_assignment + [value_res])



            


    new_factor = Factor(" " + factor1.name + " * " + factor2.name + " ", new_scope)

    new_factor.add_values(all_values)

    if multiply_verbose:
        print("\n" * 1)
        new_factor.print_table()
        print("\n" * 1)
    
    return new_factor   

    # for var in factor1.get_scope():

    #     if var not in factor2.get_scope():
    #         factor2 = restrict(factor2, var, factor1.get_scope_value(var))

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    #[[1, 2], ['a', 'b'], ['heavy', 'light']] 
    #[['heavy'] ['light'], ['this'] ['that']]
    # [1, 'a', 'heavy', "this"] [1, 'a', 'heavy', "that"]
    # [1, 'a', 'light', "this"] [1, 'a', 'light', "that"]
    # [1, 'b', 'heavy', "this"] [1, 'b', 'heavy', "that"]
    # [1, 'b', 'light', "this"] [1, 'b', 'light', "that"]
    # [2, 'a', 'heavy', "this"] [2, 'a', 'heavy', "that"]
    # [2, 'a', 'light', "this"] [2, 'a', 'light', "that"]
    # [2, 'b', 'heavy', "this"] [2, 'b', 'heavy', "that"]
    # [2, 'b', 'light', "this"] [2, 'b', 'light', "that"]

    if len(factor_list) == 0: 
        print("Factor list is empty")
        return None
    elif len(factor_list) == 1:
        return factor_list[0]
    else:
        while len(factor_list) > 1:
            factor1 = factor_list.pop(0)
            factor2 = factor_list.pop(0)
            new_factor = multiply_2_factors(factor1, factor2)
            factor_list.append(new_factor)
        return new_factor


def non_evidence_vars(factors, var_query):
    hidden_vars = set()

    for factor in factors:
        for var in factor.get_scope():
            if var != var_query:
                hidden_vars.add(var)

    return list(hidden_vars)



def restrict_evidence_variables(factors, EvindenceVars):
    for i in range(len(factors)):
        for evidence_var in EvindenceVars:
            scope = factors[i].get_scope()
            if evidence_var in scope:  
                evidence_value = evidence_var.get_evidence() 
                factors[i] = restrict(factors[i], evidence_var, evidence_value)
                if ve_verbose:
                    factors[i].print_table()

def remove_empty_factors(factors):
    for i in factors:
        if len(i.scope) == 0:
            factors.remove(i)

def pick_factors_to_multiply(factors, hidden_vars):
    for var in hidden_vars:
        if ve_verbose:
            print(" ")
            print("ELIMINATING VAR", var.name)
        factors_to_be_removed = []
        copy_factors = factors.copy()

        for i in range(len(factors)):
            current_factor = copy_factors[i]
            if var in current_factor.get_scope():
                if ve_verbose:
                    print("Factor contains hidden var", current_factor.name)
                factors_to_be_removed.append(current_factor)
                factors.remove(current_factor)  # update the list of Factors

        multiplied_factor = multiply(factors_to_be_removed)  
        if ve_verbose:
            print("             MULITIPLIED FACTOR", multiplied_factor.name)
            multiplied_factor.print_table()
        new_factor = sum_out(multiplied_factor, var) 
        if ve_verbose:
            print("             SUMMED OUT", new_factor.name)
            new_factor.print_table()
        factors.append(new_factor)  
        if ve_verbose:
            print("\n")
            print("AFTER ELIMINATION: ")
            for i in factors:
                print(i.name)
                i.print_table()

    related_factors = []
    for factor in factors:
        if (len(factor.scope) != 0) or (len(factor.scope) == 0 and factor.values[0] != 0):
            related_factors.append(factor)

    return related_factors

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
    

    factors = bayes_net.factors().copy()

    
    restrict_evidence_variables(factors, EvidenceVars)
    remove_empty_factors(factors)
    if ve_verbose:
        print("\n" * 2)
        print("     Restricted Factors To:", ", ".join([f"{var.name}={var.get_evidence()}" for var in EvidenceVars]))
        for i in factors:
            print(i.name)
            i.print_table()


    hidden_vars = non_evidence_vars(factors, var_query)
    if ve_verbose: 
        print(" ")
        print("     Hidden Vars", hidden_vars)



  

    final_factor = multiply(pick_factors_to_multiply(factors, hidden_vars))
    normalized_factor = normalize(final_factor)

    if ve_verbose:
        print("Final Factor")
        final_factor.print_table()
        print("Normalized Factor")
        normalized_factor.print_table()

    return normalized_factor


def create_variables(attributes, variable_domains):
    """Return a list of Variable objects with the name and domain instantiated"""

    variables = []  # list of Variable objects excluding Salary
    class_variable = None
    for attribute in attributes:
        new_var = Variable(attribute, variable_domains[attribute])
        if attribute == "Salary":  # use Salary as the prior
            class_variable = new_var
        else:
            variables.append(new_var)
    return variables, class_variable

def compute_conditional_prob(attribute_index, attribute_value, salary_value, dataset):
    """Given the salary_value and the attribute_value, return the conditional probability
    P(attribute=attribute_value | Salary=salary_value) by counting the frequency of the
    data that matches in dataset"""

    count_salary, count_attribute_and_salary = 0, 0

    for data in dataset:
        if data[-1] == salary_value:
            count_salary += 1
            if data[attribute_index] == attribute_value:
                count_attribute_and_salary += 1

    if count_salary == 0:
        return 0

    return count_attribute_and_salary / count_salary


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
    input_data = []
    with open('data/adult-train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    variables, class_variable = create_variables(headers, variable_domains)

    factors = []  
    for var in variables:
        factors.append(Factor('P(' + var.name + '|Salary)', [var, class_variable]))
    class_factor = Factor('P(Salary)', [class_variable])


    count_salary_less_than_50K = 0
    count_salary_more_than_50K = 0
    total = len(input_data)
    for data in input_data:
        if data[-1] == '<50K':
            count_salary_less_than_50K += 1
        elif data[-1] == '>=50K':
            count_salary_more_than_50K += 1

    class_factor.add_values([['<50K', count_salary_less_than_50K / total],
                             ['>=50K', count_salary_more_than_50K / total]])

    attribute_to_index = {}
    for index, attribute in enumerate(headers):
        attribute_to_index[attribute] = index

    for salary_value in ['<50K', '>=50K']:
        for factor in factors: 
            full_name = factor.name
            attribute = full_name[full_name.index("(") + 1:full_name.index("|Salary)")].strip()
            attribute_index = attribute_to_index[attribute]

            for attribute_value in variable_domains[attribute]:  # iterate through all the domain values of the attribute
                prob = compute_conditional_prob(attribute_index, attribute_value, salary_value, input_data)
                factor.add_values([[attribute_value, salary_value, prob]])

    return BN("Predict_Salary", variables + [class_variable], factors + [class_factor])


def get_assigned_vars(data, variables, index_dict, is_E2):

    assigned_vars = []  
    var_salary = None
    for var in variables:
        if var.name in ["Work", "Occupation", "Education", "Relationship"]:
            var_index = index_dict[var.name]
            var.set_evidence(data[var_index])
            assigned_vars.append(var)
        if is_E2:
            if var.name == "Gender":
                var_index = index_dict["Gender"]
                var.set_evidence(data[var_index])
                assigned_vars.append(var)
        if var.name == "Salary":
            var_salary = var

    return assigned_vars, var_salary


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

    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    
    attribute_to_index = {}
    for index, attribute in enumerate(headers):
        attribute_to_index[attribute] = index

    variables = bayes_net.variables()
    num_women_E1_above_E2, num_men_E1_greater_E2 = 0, 0
    women_income_prediction_count, male_income_prediction_count = 0, 0
    num_women_with_income_above50_k, count_men_with_income_above50k = 0, 0
    total_women_count, total_men_count = 0, 0
    income_percentage = 0

    for data in input_data:
        # E1 prediction
        assigned_vars, var_salary = get_assigned_vars(data, variables, attribute_to_index, is_E2=False)
        probabilityE1Factor = ve(bayes_net, var_salary, assigned_vars)
        probabilityE1 = probabilityE1Factor.values

        if probabilityE1[1] > 0.5:  # P(Salary=">=$50K"|E1)
            if data[attribute_to_index["Gender"]] == "Female":
                women_income_prediction_count += 1
                if data[attribute_to_index["Salary"]] == ">=50K":
                    num_women_with_income_above50_k += 1
            elif data[attribute_to_index["Gender"]] == "Male":
                male_income_prediction_count += 1
                if data[attribute_to_index["Salary"]] == ">=50K":
                    count_men_with_income_above50k += 1

        if question in [1, 2]:
            assigned_vars, var_salary = get_assigned_vars(data, variables, attribute_to_index, is_E2=True)
            prob_E2_factor = ve(bayes_net, var_salary, assigned_vars)
            prob_E2 = prob_E2_factor.values

            if probabilityE1[1] > prob_E2[1]:
                if data[attribute_to_index["Gender"]] == "Female":
                    num_women_E1_above_E2 += 1
                elif data[attribute_to_index["Gender"]] == "Male":
                    num_men_E1_greater_E2 += 1

        if data[attribute_to_index["Gender"]] == "Female":
            total_women_count += 1
        elif data[attribute_to_index["Gender"]] == "Male":
            total_men_count += 1

    if question == 1:
        income_percentage = num_women_E1_above_E2 / total_women_count
    elif question == 2:
        income_percentage = num_men_E1_greater_E2 / total_men_count
    elif question == 3:
        income_percentage = num_women_with_income_above50_k / women_income_prediction_count
    elif question == 4:
        income_percentage = count_men_with_income_above50k / male_income_prediction_count
    elif question == 5:
        income_percentage = women_income_prediction_count / total_women_count
    elif question == 6:
        income_percentage = male_income_prediction_count / total_men_count

    return income_percentage * 100

    

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))

