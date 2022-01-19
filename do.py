import docker
import docker.errors




def get_docker_client():

    try:
        return docker.from_env()
    except docker.errors.DockerException:
        return None


def main():

    client = get_docker_client()
    print(f'Client is {client}')

    result = client.containers.run('mlhub:latest', 'echo hello world')
    print(result)




if __name__ == '__main__':
    main()

