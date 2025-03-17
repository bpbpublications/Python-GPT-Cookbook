from prefect_email import email_check_and_queue

if __name__ == "__main__":
    email_check_and_queue.from_source(
        source='.',  # code stored in local directory
        entrypoint="prefect_local_deploy.py:email_check_and_queue",
    ).deploy(
        name="local-email-flow",
        work_pool_name="my-work-pool",
        cron="* * * * *", # Cron schedule (every minute)
        ignore_warnings=True
    )