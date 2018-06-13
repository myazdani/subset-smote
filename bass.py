
"""
Mehrdad Yazdani
MIT 
"""


import numbers
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.utils import check_random_state, safe_indexing
from sklearn.model_selection import train_test_split

from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline, Pipeline

import warnings
warnings.filterwarnings("ignore", 
  message="Function _ratio_float is deprecated; Use a float for 'ratio' \
  is deprecated from version 0.2. The support will be removed in 0.4. \
  Use a dict, str, or a callable instead.")
warnings.filterwarnings("ignore", message="From version 0.21, test_size \
  will always complement train_size unless both are specified")



class SubsetSMOTE(SMOTE):

    def __init__(self, fixed_features = None, synth_features = "auto", 
                 max_samples = 1.0, **kwargs):
        """
        Generate synthetic samples using SMOTE on only a subset of features 
        (columns). The remaining features are fixed. The features that are 
        selected for data synthesis can either be specified and randomly selected. 
        
        Parameters
        ----------          
        fixed_features: list of integers or None (default None)
            List of integers to select features to keep fixed. If None (defulat), then a random
            subset of features will be fixed according to synth_features. The union of 
            fixed_features and synth_features equals the entire feature set of the data.         
        
        synth_features:  int, float, string, list of integers, or None, optional (default="auto") 
            Features to use for creating synthetic examples. Remaining features kept fixed and not used for 
            synthetic data creation. Ignored if fixed_features is not None. 

            - If int, then randomly draw synth_features number of features to use for synthetic data creation.
            - If float, then randomly draw the int(synth_features * n_features) percentage of features for 
            synthetic data creation.
            - If "auto", then synth_features=sqrt(n_features).
            - If "sqrt", then synth_features=sqrt(n_features) (same as "auto").
            - If "log2", then synth_features=log2(n_features).
            - If None, then synth_features is the remaining features after removing fixed_features.
            
        max_samples : float (default 1.0)
            Number of stratified samples to draw from X. This is if you want to under-sample the data at 
            random first (to prevent getting data to be too big). Useful for creating bagged ensembles.
            
        **kwargs:
            Additional parameters for SMOTE
        
        Example
        -------
        >> from imblearn.over_sampling import SMOTE
        >> from BASS import feature_smote
        >> # load imabalanced data: X (features), y (targets)
        >> sampler = feature_smote(combination_cols=var_cols, control_cols=fix_cols, kind = "regular") 
        >> num_features = X.shape[1]
        >> shuffled_cols = np.array(random.sample(range(num_features), num_features))
        >> num_features_to_interpolate = int(np.sqrt(num_features)) # using sqrt as suggested by Breiman 
        >> fix_cols = shuffled_cols[:num_features_to_interpolate]
        >> var_cols = shuffled_cols[num_features_to_interpolate:]
        >> sampler = feature_smote(combination_cols=var_cols, control_cols=fix_cols, kind = 'regular')
        >> X_bal, y_bal = sampler.fit_sample(X, y) # this is just one bag of balanced data. 
        >> # iteretate process to generate multiple bags
        """
        super().__init__(**kwargs)   
        
        self.synth_features = synth_features
        self.fixed_features = fixed_features
        self.max_samples = max_samples

        
    def _make_samples(self, X, y_type,nn_data, nn_num, n_samples, step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.
        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.
        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used
        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in nn_data.
        n_samples : int
            The number of samples to generate.
        step_size : float, optional (default=1.)
            The step size to create samples.
        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.
        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.


        TO DO:
        ------
            Add sparse matrix support

        """
        
        if self.max_samples < 1.0:
            # randomly sample data according with size max_samples (float percentage)
            X, _, y_type, _ = train_test_split(X, y_type, train_size = self.max_samples, 
                                               stratify = y_type)
        

        num_features = X.shape[1]
        if self.synth_features == "auto":
            # take sqrt of number of features to randomly synthesize
            num_synth_features = np.int(np.sqrt(num_features))
        else:
            # take self.synth_features percentage of the features
            # to randomly synthesize
            num_synth_features = np.int(self.synth_features*num_features)
            
        
        ## randomly select the features according the num_synth_features
        self.synth_cols = np.random.choice(range(num_features), 
                                      size = num_synth_features, 
                                      replace=False)

        # get indexes of all features
        all_cols = np.arange(num_features)            


        # figure out which columns should not be used for data synthesis, ie 
        # remain fixed 
        self.fixed_cols = np.array(list(set(all_cols) - set(all_cols[self.synth_cols])))
            
        
        
        ##
        # SMOTE params setup
        ##
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)
        #steps = 1e6 * random_state.uniform(size=n_samples)
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        

        '''
        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                if X[row].nnz:
                    sample = X[row] - step * (X[row] -
                                              nn_data[nn_num[row, col]])
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()
        else:
        '''
        X_new = np.zeros((n_samples, X.shape[1]))
        for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
            # only apply SMOTE data synthesis to synth_cols
            X_new[i,self.synth_cols] = (X[row,self.synth_cols] - 
                                            step * (X[row, self.synth_cols] - 
                                            nn_data[nn_num[row, col], self.synth_cols]))
            # don't change fixrd columns
            X_new[i,self.fixed_cols] = X[row,self.fixed_cols]

        y_new = np.array([y_type] * len(samples_indices))
        '''
        if sparse.issparse(X):
            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [len(samples_indices), X.shape[1]]),
                    y_new)
        else:
        '''
        return X_new, y_new


class RandomSampler(BaseUnderSampler):
    """Class to perform random under-sampling.
    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.
    Read more in the :ref:`User Guide <controlled_under_sampling>`.
    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.
        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.
    replacement : boolean, optional (default=False)
        Whether the sample is with or without replacement.
    Notes
    -----
    Supports mutli-class resampling by sampling each class independently.
    See
    :ref:`sphx_glr_auto_examples_plot_ratio_usage.py` and
    :ref:`sphx_glr_auto_examples_under-sampling_plot_random_under_sampler.py`
    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})
    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False):
        '''
        super(RandomSampler, self).__init__(
            ratio=ratio, random_state=random_state)
        '''
        super(RandomSampler, self).__init__(random_state=random_state)        
        self.return_indices = return_indices
        self.replacement = replacement
        self.ratio = ratio

    def _sample(self, X, y):
        """Resample the dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.
        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, an array will be returned
            containing a boolean for each sample to represent whether
            that sample was selected or not.
        """
        random_state = check_random_state(self.random_state)
        
        '''
        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under, np.flatnonzero(y == target_class)[
                    index_target_class]), axis=0)
        '''
        indices, _, = train_test_split(np.arange(len(y)), test_size = 1.0 - self.ratio, 
                                       stratify = y, random_state = random_state)
        
        if self.return_indices:
            return (safe_indexing(X, indices), safe_indexing(y, indices),
                    indices)
        else:
            return safe_indexing(X, indices), safe_indexing(y, indices)

class SyntheticBaggingClassifier(BaggingClassifier):
    """A Bagging classifier with additional balancing using SMOTE or SubsetSMOTE.
    This implementation of Bagging is similar to the scikit-learn
    implementation. It includes an additional step to balance the training set
    at fit time using a ``RandomSampler`` followed by ``SMOTE`` or ``SubsetSMOTE``
    Read more in the :ref:`User Guide <ensemble_meta_estimators>`.
    
    
    TODO:
    -----
    Add more details about the relevant params.
    
    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.
    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        - If int, then draw ``max_samples`` samples.
        - If float, then draw ``max_samples * X.shape[0]`` samples.
    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        - If int, then draw ``max_features`` features.
        - If float, then draw ``max_features * X.shape[1]`` features.
    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.
    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.
    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.
    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
        .. versionadded:: 0.17
           *warm_start* constructor parameter.
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.
        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.
    replacement : bool, optional (default=False)
        Whether or not to sample randomly with replacement or not.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random
          number generator;
        - If ``None``, the random number generator is the
          ``RandomState`` instance used by ``np.random``.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.
    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.
    classes_ : array, shape (n_classes,)
        The classes labels.
    n_classes_ : int or list
        The number of classes.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : ndarray, shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        ``oob_decision_function_`` might contain NaN.
    """
    def __init__(self,
                 synth_sampling = SubsetSMOTE(),
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 ratio='auto',
                 replacement=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super(BaggingClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.ratio = ratio
        self.replacement = replacement
        self.synth_sampling = synth_sampling

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "gotb {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)
        '''
        self.base_estimator_ = Pipeline(
            [('sampler', RandomUnderSampler(ratio=self.ratio,
                                            replacement=self.replacement)),
             ('classifier', base_estimator)])
        '''                 

        self.base_estimator_ = Pipeline([('sampler', RandomSampler(ratio=self.ratio)), 
                                         ('smote', self.synth_sampling),
                                         ('classifier', base_estimator)])

    def fit(self, X, y):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        Returns
        -------
        self : object
            Returns self.
        """
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, sample_weight=None)