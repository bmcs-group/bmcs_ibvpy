
from traits.api import provides, HasTraits

from .i_sdomain import ISDomain


@provides(ISDomain)
class SDomain(HasTraits):
    pass
