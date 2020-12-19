
from traits.api import \
    HasStrictTraits, \
    List, \
    on_trait_change, Property, cached_property, \
    Event
import numpy as np


class DomainStateContainer(HasStrictTraits):
    '''Model of the spatial domain - base class approximations and discretizations.

    The XModel represents one subdomain within a spatial domain.
    '''

    def __init__(self, subdomains, *args, **kw):
        super().__init__(*args, **kw)
        self.subdomains = subdomains
        self.serialized_subdomains

    changed_structure = Event

    subdomains = List(domain_changed=True)

    @on_trait_change('changed_structure')
    def _validate_subdomains(self):
        for domain in self.subdomains:
            domain.validate()

    serialized_subdomains = Property(depends_on='subdomains, subdomains_items')

    @cached_property
    def _get_serialized_subdomains(self):
        '''Link the new subdomain at the end of the series.
        '''
        s = np.array(self.subdomains)
        for s1, s2 in zip(s[:-1], s[1:]):
            s1.xmodel.set_next(s2.xmodel)
            s2.xmodel.set_prev(s1.xmodel)
        return self.subdomains

    nonempty_subdomains = Property(depends_on='changed_structure')

    @cached_property
    def _get_nonempty_subdomains(self):
        d_list = []
        for d in self.serialized_subdomains:
            if d.xmodel.n_active_elems > 0:
                d_list.append(d)
        return d_list

    n_dofs = Property

    def _get_n_dofs(self):
        '''Return the total number of dofs in the domain.
        Use the last subdomain's: dof_offset + n_dofs 
        '''
        last_d = self.serialized_subdomains[-1]
        dof_offset = last_d.xmodel.dof_offset
        n_dofs = last_d.xmodel.n_dofs
        return dof_offset + n_dofs

    dof_offset_arr = Property

    def _get_dof_offset_arr(self):
        '''
        Return array of the dof offsets 
        from serialized subdomains
        '''
        a = np.array([domain.xmodel.dof_offset
                      for domain in self.serialized_subdomains])
        return a

    U_var_shape = Property

    def _get_U_var_shape(self):
        return (self.n_dofs,)

    def __iter__(self):
        return iter(self.subdomains)

    def __getitem__(self, idx):
        return self.serialized_subdomains[idx]
