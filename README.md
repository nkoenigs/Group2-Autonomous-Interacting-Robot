# zGroup2
group 2 master


code changes for cvController!
elif commandFromQueue == "aprilFollow":
                # We know the next 2 items in the queue in this case are the x and z coordinates  - grab them
                # Note: get() commands will block until it can get something
                cvQueue.put("Receive april Tag request")
                target_tag_number = cvQueue.get()
                target_tag_radius = cvQueue.get()
                cvObject.april_following(target_tag_number, target_tag_radius, cvQueue)
                cvQueue.task_done()
		cvQueue.task_done()