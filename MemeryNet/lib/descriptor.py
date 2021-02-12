import numpy as np

class s_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._s
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: s, size=(n,1)
        :return: matrics['s'], size=(sentence, n, 1)
        """
        obj._s = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy() # (1,n,1)
            if 's' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['s'] = np.concatenate((obj.matrics[obj.story_index]['s'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['s'] = vector
            obj.state_path[obj.story_index].append('s')

class G_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._G
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: G, size=(1, m)
        :return: matrics['G'], size=(sentence, 1, m)
        """
        obj._G = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'G' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['G'] = np.concatenate((obj.matrics[obj.story_index]['G'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['G'] = vector
            obj.state_path[obj.story_index].append('G')

class H_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._H
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: H, size=(n, m)
        :return: matrics['H'], size=(sentence, n, m)
        """
        obj._H = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'H' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['H'] = np.concatenate((obj.matrics[obj.story_index]['H'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['H'] = vector
            obj.state_path[obj.story_index].append('H')

class new_H_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._new_H
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: new_H, size=(n, m)
        :return: matrics['new_H'], size=(sentence, n, m)
        """
        obj._new_H = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'new_H' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['new_H'] = np.concatenate((obj.matrics[obj.story_index]['new_H'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['new_H'] = vector
            obj.state_path[obj.story_index].append('new_H')

class q_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._q
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: q, size=(n, 1)
        :return: matrics['q'], size=(sentence, n, 1)
        """
        obj._q = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'q' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['q'] = np.concatenate((obj.matrics[obj.story_index]['q'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['q'] = vector
            obj.state_path[obj.story_index].append('q')

class p_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._p
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: p, size=(1, m)
        :return: matrics['p'], size=(sentence, 1, m)
        """
        obj._p = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'p' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['p'] = np.concatenate((obj.matrics[obj.story_index]['p'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['p'] = vector
            obj.state_path[obj.story_index].append('p')

class u_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._u
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: u, size=(n, 1)
        :return: matrics['u'], size=(sentence, n, 1)
        """
        obj._u = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'u' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['u'] = np.concatenate((obj.matrics[obj.story_index]['u'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['u'] = vector
            obj.state_path[obj.story_index].append('u')

class ans_vector_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._ans_vector
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: ans_vector, size=(m, 1)
        :return: matrics['ans_vector'], size=(sentence, m, 1)
        """
        obj._ans_vector = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'ans_vector' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['ans_vector'] = np.concatenate((obj.matrics[obj.story_index]['ans_vector'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['ans_vector'] = vector
            obj.state_path[obj.story_index].append('ans_vector')

class ans_AttriAccess:

    def __get__(self, obj, objtype=None):
        value = obj._ans
        return value

    def __set__(self, obj, value):
        """
        :param obj: entNet
        :param value: ans, size=(m, 1)
        :return: matrics['ans'], size=(sentence, m, 1)
        """
        obj._ans = value
        if obj.record_allowed:
            vector = value.detach().cpu().unsqueeze(dim=0).numpy()
            if 'ans' in obj.matrics[obj.story_index]:
                obj.matrics[obj.story_index]['ans'] = np.concatenate((obj.matrics[obj.story_index]['ans'], vector), axis=0)
            else:
                obj.matrics[obj.story_index]['ans'] = vector
            obj.state_path[obj.story_index].append('ans')
            obj.state_path[obj.story_index].append('grad')