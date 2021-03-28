package com.ssafy.authsvr.entity;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.ssafy.authsvr.payload.InfoUpdateRequest;
import lombok.Getter;
import lombok.Setter;
import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;


@Getter
@Setter
@Document(collection = "user")
public class User {

    private ObjectId id;

    private String name; // nickname

    private String email;

    private String imageUrl;

    @JsonIgnore
    private String password;

    private AuthProvider provider;

    private String providerId;

    private List<Integer> categoryList;
    private List<Integer> blogList;
    private List<Integer> youtubeList;
    private List<Integer> jobList;

    public void update(InfoUpdateRequest req){
        Optional.ofNullable(req.getName()).ifPresent((x)-> this.name = req.getName());
        Optional.ofNullable(req.getCategoryList()).ifPresent((x)-> this.categoryList = req.getCategoryList());
        Optional.ofNullable(req.getBlogList()).ifPresent((x)-> this.blogList = req.getBlogList());
        Optional.ofNullable(req.getYoutubeList()).ifPresent((x)-> this.youtubeList = req.getYoutubeList());
        Optional.ofNullable(req.getJobList()).ifPresent((x)-> this.jobList = req.getJobList());
    }
}
