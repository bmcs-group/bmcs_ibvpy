#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Aug 7, 2009 by: rchx
'''
A trait type with shadow attribute that allows database referencing -
database keys are allowed as values, the shadow attribute is then the
appropriate database member.
'''

from traits.api import \
    TraitType, HasStrictTraits, TraitError, \
    Property, Button

class KeyRef(TraitType):

    is_mapped = True

    def __init__(self, *args, **metadata):
        if len(args) == 1:
            self._default_key = args[0]
        else:
            self._default_key = None
        db = metadata.get('db', None)
        if db:
            self.map = db
        else:
            raise ValueError('db not defined')

        super(KeyRef, self).__init__(**metadata)

    def validate(self, obj, name, key):
        ''' Set the trait value '''

        self.keys_name = name + '_keys'
        keys_prop = Property(fget=lambda: list(self.map.keys()))
        obj.add_trait(self.keys_name, keys_prop)

        if key in list(self.map.keys()):
            prechange_mapped_val = getattr(obj, name + '_')
            try:
                prechange_mapped_val.del_link(obj)
                '''Management of backward links - erase the link to an object that has
                stopped using me.
                '''
            except AttributeError:
                print(('Warning: Unable to remove reference to object from '
                'mapped value of type %s (Backwards link management not implemented'
                ' in mapped type), trait name %s') % (prechange_mapped_val.__class__, name))
            postchange_mapped_val = self.mapped_value(key)
            try:
                postchange_mapped_val.add_link(obj)
                '''Management of backward links - add the link to an object that has
                started using me.
                '''
            except AttributeError:
                print(('Warning: Unable to pass reference to object to mapped '
                'value of type %s (Backwards link management not implemented '
                'in mapped type)') % prechange_mapped_val.__class__)

            self.post_setattr(obj, name, key)
            ''' When the new value is not different from the old,
            post_setattr does not get called, which can cause issues
            '''
            return key
        else:
            self.error(object, name, key)

    def get_default_value(self):
        '''Take the default value'''

        keys = list(self.map.keys())
        if self._default_key == None:
            if len(keys) > 1:
                return (0, list(self.map.keys())[0])
            else:
                return (0, None)
                raise TraitError('invalid key, no entries in database extension %s' % \
                    (self.map.klass,))
        if self._default_key in keys:
            return (0, self._default_key)

        raise TraitError('assigned default value must be one of %s but a value of %s was received' % \
            (list(self.map.keys()), self._default_key))

    def mapped_value (self, key):
        try:
            return self.map[ key ]
        except:
            return None

    def post_setattr (self, object, name, value):
        val = self.mapped_value(value)
        try:
            setattr(object, name + '_', val)
        except:
            # We don't need a fancy error message, because this exception
            # should always be caught by a TraitCompound handler:
            raise TraitError('Unmappable')

    def info (self):
        keys = [ repr(x) for x in list(self.map.keys()) ]
        keys.sort()
        return ' or '.join(keys)

    def get_editor (self, trait=None):
        return self.create_editor()

    def create_editor(self):
        pass

