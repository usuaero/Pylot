#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 textureCoords;
layout(location = 2) in vec3 vertNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 orientation;

out vec2 newTexture;
out vec3 fragNormal;

void main()
{
	fragNormal = (orientation*vec4(vertNormal, 0.0f)).xyz;
    gl_Position = proj * view * model *orientation* vec4(position, 1.0f);
    newTexture = vec2(textureCoords.x, 1 - textureCoords.y);
}
