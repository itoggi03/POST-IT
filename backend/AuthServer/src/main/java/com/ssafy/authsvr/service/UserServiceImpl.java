package com.ssafy.authsvr.service;


import com.ssafy.authsvr.entity.User;
import com.ssafy.authsvr.payload.InfoUpdateRequest;
import com.ssafy.authsvr.repository.UserRepository;
import lombok.AllArgsConstructor;
import org.bson.types.ObjectId;
import org.springframework.stereotype.Service;



@Service
@AllArgsConstructor
public class UserServiceImpl implements UserService {

    UserRepository userRepository;

    @Override
    public User getCurrentUser(ObjectId id) {
        return userRepository.findById(id).get();
    }

    @Override
    public User updateCurrentUser(ObjectId id, InfoUpdateRequest infoUpdateRequest) {
        User user = userRepository.findById(id).get();
        if(user != null)
            return userRepository.save(user.update(infoUpdateRequest));
        else
            return null;
    }

}
